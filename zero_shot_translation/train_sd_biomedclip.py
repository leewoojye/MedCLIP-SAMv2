"""
SD Inpainting + BiomedCLIP Cross-Attention Fine-tuning

Replaces SD's CLIP text encoder with finetuned BiomedCLIP (DHN),
then trains only U-Net Cross-Attention K_proj and V_proj weights
so the U-Net learns to interpret BiomedCLIP embeddings.

Training uses standard diffusion denoising loss on MedPix image-caption pairs.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from diffusers.models.attention_processor import Attention
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import csv
import os
import argparse
import random
import numpy as np
from tqdm import tqdm


# ============================================================
# 1. Dataset
# ============================================================
class MedPixInpaintDataset(Dataset):
    """MedPix image-caption pairs with random mask generation."""

    def __init__(self, csv_path, image_root, image_size=512):
        self.image_size = image_size
        self.pairs = []

        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                caption = row.get("Caption", "").strip()
                fname = row.get("filename", "").strip()
                if not caption or len(caption) < 5:
                    continue
                # CSV has paths like "data/medpix_dataset/images/synpicXXX.jpg"
                # Convert to absolute path relative to image_root
                img_basename = os.path.basename(fname)
                img_path = os.path.join(image_root, img_basename)
                if os.path.exists(img_path):
                    self.pairs.append((img_path, caption))

        print(f"Loaded {len(self.pairs)} valid image-caption pairs")

    def __len__(self):
        return len(self.pairs)

    def _random_mask(self):
        """Generate a random rectangular mask."""
        mask = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        # Random rectangle (20-60% of image)
        h_ratio = random.uniform(0.2, 0.6)
        w_ratio = random.uniform(0.2, 0.6)
        h = int(self.image_size * h_ratio)
        w = int(self.image_size * w_ratio)
        top = random.randint(0, self.image_size - h)
        left = random.randint(0, self.image_size - w)
        mask[top : top + h, left : left + w] = 1.0

        return torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)

    def __getitem__(self, idx):
        img_path, caption = self.pairs[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback to a random valid index
            return self.__getitem__(random.randint(0, len(self.pairs) - 1))

        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)

        # Convert to tensor [-1, 1]
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)
        image = image * 2.0 - 1.0

        mask = self._random_mask()  # (1, H, W)

        return {"image": image, "mask": mask, "caption": caption}


# ============================================================
# 2. BiomedCLIP Text Encoder Wrapper
# ============================================================
class BiomedCLIPTextEncoder(torch.nn.Module):
    """Wraps finetuned BiomedCLIP to produce text embeddings compatible with SD U-Net."""

    def __init__(self, model_path, device="cuda"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True
        )
        if not hasattr(self.model.config.text_config, "is_decoder"):
            self.model.config.text_config.is_decoder = False
        self.model.eval()
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, texts, device="cuda"):
        """Encode texts to hidden states (batch, seq_len, 768)."""
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        vocab_size = self.model.config.text_config.vocab_size
        if inputs["input_ids"].max() >= vocab_size:
            inputs["input_ids"] = torch.clamp(inputs["input_ids"], max=vocab_size - 1)
        inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
        seq_len = inputs["input_ids"].shape[1]
        batch_size = inputs["input_ids"].shape[0]
        inputs["position_ids"] = (
            torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            text_outputs = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"],
                position_ids=inputs["position_ids"],
                output_hidden_states=True,
            )
        return text_outputs[0]  # (batch, seq_len, 768)


# ============================================================
# 3. Utility: freeze/unfreeze specific parameters
# ============================================================
def setup_trainable_params(unet):
    """
    Freeze entire U-Net, then unfreeze only Cross-Attention K_proj and V_proj.
    Returns list of trainable parameters.
    """
    # Freeze everything
    for param in unet.parameters():
        param.requires_grad = False

    trainable_params = []
    trainable_names = []

    for name, module in unet.named_modules():
        # Cross-attention modules contain "attn2" in their name (attn1 = self-attn)
        if "attn2" in name:
            for pname, param in module.named_parameters(recurse=False):
                if "to_k" in pname or "to_v" in pname:
                    param.requires_grad = True
                    trainable_params.append(param)
                    trainable_names.append(f"{name}.{pname}")

    # If attn2 naming didn't work, try alternative naming patterns
    if not trainable_params:
        for name, param in unet.named_parameters():
            if ("attn2" in name or "cross_attn" in name) and (
                "to_k" in name or "to_v" in name
            ):
                param.requires_grad = True
                trainable_params.append(param)
                trainable_names.append(name)

    print(f"Trainable parameters: {len(trainable_names)}")
    total = sum(p.numel() for p in trainable_params)
    print(f"Total trainable: {total:,} ({total / 1e6:.2f}M)")
    for n in trainable_names[:5]:
        print(f"  {n}")
    if len(trainable_names) > 5:
        print(f"  ... and {len(trainable_names) - 5} more")

    return trainable_params


# ============================================================
# 4. Training Loop
# ============================================================
def train(args):
    device = torch.device(args.device)

    # --- Load SD Inpainting Pipeline ---
    print("Loading SD Inpainting pipeline...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.sd_model,
        torch_dtype=torch.float32,  # Train in fp32 for stability
    )

    unet = pipe.unet.to(device)
    vae = pipe.vae.to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.sd_model, subfolder="scheduler"
    )

    # Freeze VAE
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False

    # --- Load BiomedCLIP Text Encoder ---
    print(f"Loading BiomedCLIP from: {args.biomedclip_path}")
    text_encoder = BiomedCLIPTextEncoder(args.biomedclip_path, device)
    text_encoder = text_encoder.to(device)

    # --- Setup trainable params (Cross-Attention K,V only) ---
    trainable_params = setup_trainable_params(unet)

    if not trainable_params:
        raise RuntimeError("No trainable parameters found! Check U-Net architecture.")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # --- Dataset ---
    dataset = MedPixInpaintDataset(
        csv_path=args.medpix_csv,
        image_root=args.medpix_images,
        image_size=args.image_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # --- Training ---
    os.makedirs(args.output_dir, exist_ok=True)
    global_step = 0
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        unet.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            images = batch["image"].to(device)       # (B, 3, H, W) [-1,1]
            masks = batch["mask"].to(device)          # (B, 1, H, W) [0,1]
            captions = batch["caption"]               # list of strings

            # 1. Encode images to latent space
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * vae.config.scaling_factor  # (B, 4, H/8, W/8)

            # 2. Prepare masked latent (for inpainting)
            masked_images = images * (1 - masks)  # zero out masked area
            with torch.no_grad():
                masked_latents = vae.encode(masked_images).latent_dist.sample()
                masked_latents = masked_latents * vae.config.scaling_factor

            # Resize mask to latent size
            mask_latent = F.interpolate(
                masks, size=latents.shape[-2:], mode="nearest"
            )

            # 3. Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 4. Get text embeddings from BiomedCLIP
            encoder_hidden_states = text_encoder.encode(captions, device=device)
            # Cast to float32 for training
            encoder_hidden_states = encoder_hidden_states.float()

            # 5. Prepare inpainting input (9 channels)
            inpaint_input = torch.cat(
                [noisy_latents, mask_latent, masked_latents], dim=1
            )

            # 6. Predict noise
            noise_pred = unet(
                inpaint_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

            # 7. Compute loss
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % args.log_every == 0 or epoch == 1:
            print(
                f"Epoch {epoch}/{args.epochs} | "
                f"Avg Loss: {avg_loss:.6f} | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

        # Save best and periodic checkpoints
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_cross_attn_weights(unet, os.path.join(args.output_dir, "best.pt"))

        if epoch % args.save_every == 0:
            save_cross_attn_weights(
                unet, os.path.join(args.output_dir, f"epoch_{epoch}.pt")
            )

    # Final save
    save_cross_attn_weights(unet, os.path.join(args.output_dir, "final.pt"))
    print(f"\nTraining complete. Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved to: {args.output_dir}")


def save_cross_attn_weights(unet, path):
    """Save only the trainable Cross-Attention K,V weights."""
    state = {}
    for name, param in unet.named_parameters():
        if param.requires_grad:
            state[name] = param.data.cpu()
    torch.save(state, path)
    print(f"Saved {len(state)} parameter tensors to {path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fine-tune SD Inpainting Cross-Attention with BiomedCLIP"
    )
    parser.add_argument(
        "--sd-model", type=str, default="runwayml/stable-diffusion-inpainting"
    )
    parser.add_argument(
        "--biomedclip-path", type=str, default="./saliency_maps/model"
    )
    parser.add_argument(
        "--medpix-csv",
        type=str,
        default="./biomedclip_finetuning/open_clip/src/data/medpix_dataset/medpix_dataset_clean.csv",
    )
    parser.add_argument(
        "--medpix-images",
        type=str,
        default="./biomedclip_finetuning/open_clip/src/data/medpix_dataset/images",
    )
    parser.add_argument("--output-dir", type=str, default="./zero_shot_translation/sd_biomedclip_ckpt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    train(args)
