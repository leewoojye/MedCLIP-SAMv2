import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import math

# --- 1. Dataset for Text Disentanglement ---
class TumorToHealthyDataset(Dataset):
    def __init__(self, tumor_dir, healthy_dir, mask_dir, tokenizer, size=512):
        self.tumor_dir = tumor_dir
        self.healthy_dir = healthy_dir
        self.mask_dir = mask_dir
        
        self.tokenizer = tokenizer
        self.size = size
        
        self.samples = []
        
        # We expect files named like "000001.png" in all three directories
        for f in os.listdir(tumor_dir):
            if f.endswith(".png"):
                tumor_path = os.path.join(tumor_dir, f)
                healthy_path = os.path.join(healthy_dir, f)
                mask_path = os.path.join(mask_dir, f)
                
                # Verify all three exist
                if os.path.exists(healthy_path) and os.path.exists(mask_path):
                    self.samples.append((tumor_path, healthy_path, mask_path))
        
        print(f"Loaded {len(self.samples)} valid training triplets.")

        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

        # Natural language targeting
        self.prompt = "a healthy breast"
        inputs = self.tokenizer(
            self.prompt, padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt"
        )
        self.prompt_input_ids = inputs.input_ids[0]
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        tumor_path, healthy_path, mask_path = self.samples[idx]
        
        # Original Image (with tumor)
        image = Image.open(tumor_path).convert("RGB")
        # Target Image (Healthy)
        target = Image.open(healthy_path).convert("RGB")
        # Mask Image (1s on tumor, 0s elsewhere)
        mask = Image.open(mask_path).convert("L")

        image_tensor = self.transform(image)
        target_tensor = self.transform(target)
        mask_tensor = self.mask_transform(mask)
        
        # Values < 0.5 are 0, > 0.5 are 1
        mask_tensor = torch.where(mask_tensor < 0.5, 0.0, 1.0)

        # In SD inpainting, the model takes `masked_image` as input
        # Masked image is original image where tumor mask region is blacked out or zeroed
        masked_image_tensor = image_tensor * (mask_tensor < 0.5)

        return {
            "target_pixel_values": target_tensor,
            "masked_image_values": masked_image_tensor,
            "mask_values": mask_tensor,
            "input_ids": self.prompt_input_ids,
        }


# --- 2. Main Training Loop ---
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading Base Model: {args.pretrained_model_name_or_path}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.float32, 
        safety_checker=None
    )
    
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder.to(device)
    vae = pipe.vae.to(device)
    unet = pipe.unet.to(device)
    noise_scheduler = pipe.scheduler
    
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
        
    dataset = TumorToHealthyDataset(args.tumor_dir, args.healthy_dir, args.mask_dir, tokenizer)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
    # 2.2 Inject LoRA into U-Net
    print("Injecting LoRA parameters into U-Net Cross Attention Layers...")
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"], 
    )
    # Different diffusers/peft versions expect different signatures.
    # Pass lora_config as the single argument to target the model correctly.
    unet.add_adapter(lora_config)
    unet.train()
    
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    optimizer = torch.optim.AdamW(lora_layers, lr=args.learning_rate)
    
    # Calculate training steps
    num_update_steps_per_epoch = len(dataloader)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler("constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=max_train_steps)

    print(f"Starting Training! ({args.num_epochs} Epochs)")
    progress_bar = tqdm(range(max_train_steps), desc="Steps")
    global_step = 0
    
    for epoch in range(args.num_epochs):
        for step, batch in enumerate(dataloader):
            # Move to device
            target_pixel_values = batch["target_pixel_values"].to(device)
            masked_image_values = batch["masked_image_values"].to(device)
            mask_values = batch["mask_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            # Encode images to latent space
            with torch.no_grad():
                latents = vae.encode(target_pixel_values).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                masked_image_latents = vae.encode(masked_image_values).latent_dist.sample()
                masked_image_latents = masked_image_latents * vae.config.scaling_factor
                
                mask_latents = F.interpolate(mask_values, size=(latents.shape[2], latents.shape[3]))
            
            # Sample noise to add to the latents (target healthy latents)
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Concatenate for Inpainting: (Noisy Target, Mask, Masked Image)
            # The U-Net expects 9 channels: 4 (noisy latents) + 1 (mask) + 4 (masked image latents)
            latent_model_input = torch.cat([noisy_latents, mask_latents, masked_image_latents], dim=1)

            # Get the text embedding for conditioning
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]

            # Predict the noise residual
            model_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample

            # Compute Loss (MSE between predicted noise and actual noise)
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)
            global_step += 1
            
            if global_step % 10 == 0:
                progress_bar.set_postfix(loss=loss.item())

    # Save the custom LoRA layers
    print(f"Saving Text Disentanglement LoRA to {args.output_dir}")
    unet.save_pretrained(args.output_dir)
    # Important: Save the whole pipeline so the new token embedding is saved alongside the text encoder
    pipe.save_pretrained(os.path.join(args.output_dir, "pipeline"))
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tumor_dir", type=str, required=True, help="Directory with original tumor images")
    parser.add_argument("--healthy_dir", type=str, required=True, help="Directory with healed images")
    parser.add_argument("--mask_dir", type=str, required=True, help="Directory with tumor masks")
    parser.add_argument("--output_dir", type=str, default="zero_shot_translation/lora_weights")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    args = parser.parse_args()
    main(args)
