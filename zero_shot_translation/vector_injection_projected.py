"""
Phase 3: Vector Injection with Projection Layer.

Uses the trained projector to convert BiomedCLIP shift vectors into SD CLIP space
before injecting into Stable Diffusion's U-Net Cross-Attention.

Key difference from vector_injection_breast.py:
- Shift is computed in finetuned BiomedCLIP space (medically accurate direction)
- Projected to SD CLIP space via trained linear layer
- SD base embedding provides the anchor in SD's native space
- Result: SD correctly interprets the shift in Cross-Attention K,V
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionInpaintPipeline
from transformers import AutoTokenizer, AutoModel, CLIPTokenizer, CLIPTextModel
import argparse
from PIL import Image, ImageFilter
import os


class EmbeddingProjector(nn.Module):
    """Linear projection: BiomedCLIP (768) → SD CLIP (768)."""

    def __init__(self, hidden_dim=768):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.proj(x)


def load_finetuned_biomedclip(model_path, device="cuda"):
    """Load MedCLIP-SAMv2 finetuned BiomedCLIP."""
    print(f"Loading Finetuned BiomedCLIP from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        "chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
    if not hasattr(model.config.text_config, "is_decoder"):
        model.config.text_config.is_decoder = False
    model.eval()
    return tokenizer, model


def get_biomedclip_hidden(tokenizer, model, texts, device="cuda", target_len=77):
    """Get BiomedCLIP text hidden states, truncated/padded to target_len."""
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    vocab_size = model.config.text_config.vocab_size

    if inputs["input_ids"].max() >= vocab_size:
        inputs["input_ids"] = torch.clamp(inputs["input_ids"], max=vocab_size - 1)

    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    seq_length = inputs["input_ids"].shape[1]
    batch_size = inputs["input_ids"].shape[0]
    inputs["position_ids"] = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            output_hidden_states=True,
        )
        hidden = text_outputs[0]

    if hidden.shape[1] > target_len:
        hidden = hidden[:, :target_len, :]
    elif hidden.shape[1] < target_len:
        pad_size = target_len - hidden.shape[1]
        hidden = F.pad(hidden, (0, 0, 0, pad_size), "constant", 0)

    return hidden  # (batch, 77, 768)


def get_sd_text_embedding(pipe, text, device="cuda"):
    """Get SD CLIP text embedding using the pipeline's text encoder."""
    inputs = pipe.tokenizer(
        text, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = pipe.text_encoder(**inputs)
        hidden = outputs.last_hidden_state  # (1, 77, 768)

    return hidden


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Compute shift vector in finetuned BiomedCLIP space
    print("\n--- 1. Computing Semantic Shift in Finetuned BiomedCLIP Space ---")
    bio_tokenizer, bio_model = load_finetuned_biomedclip(args.biomedclip_path, device)

    print(f"Anchors: '{args.tumor_anchor}' → '{args.healthy_anchor}'")
    bio_embs = get_biomedclip_hidden(
        bio_tokenizer, bio_model, [args.tumor_anchor, args.healthy_anchor], device
    )
    bio_tumor = bio_embs[0:1]  # (1, 77, 768)
    bio_healthy = bio_embs[1:2]

    shift_bio = bio_healthy - bio_tumor  # "Healing shift" in BiomedCLIP space
    print(f"Shift vector norm: {shift_bio.norm():.4f}")

    # Free BiomedCLIP memory
    del bio_model, bio_tokenizer
    torch.cuda.empty_cache()

    # 2. Load projector
    print("\n--- 2. Loading Projection Layer ---")
    projector = EmbeddingProjector(768).to(device)
    projector.load_state_dict(torch.load(args.projector_path, map_location=device))
    projector.eval()

    # Project shift vector to SD space
    with torch.no_grad():
        shift_sd = projector(shift_bio)  # (1, 77, 768) in SD CLIP space
    print(f"Projected shift norm: {shift_sd.norm():.4f}")

    # Free projector memory
    del projector
    torch.cuda.empty_cache()

    # 3. Load SD Inpainting Pipeline
    print(f"\n--- 3. Loading SD Inpainting ({args.sd_model}) ---")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.sd_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    pipe.safety_checker = None

    # 4. Compute target embedding in SD's native space
    print("\n--- 4. Computing Target Embedding ---")
    sd_base = get_sd_text_embedding(pipe, args.tumor_anchor, device)  # SD's understanding of "tumor breast"
    sd_uncond = get_sd_text_embedding(pipe, "", device)  # Unconditional

    alpha = args.alpha
    target_embedding = sd_base + alpha * shift_sd.to(dtype=sd_base.dtype)
    print(f"Alpha: {alpha}, Target embedding shape: {target_embedding.shape}")

    # 5. Load and process image
    print(f"\n--- 5. Processing Image ---")
    image = Image.open(args.image).convert("RGB")
    mask = Image.open(args.mask).convert("L")

    if args.dilate_mask > 0:
        print(f"Dilating mask by {args.dilate_mask} iterations...")
        for _ in range(args.dilate_mask):
            mask = mask.filter(ImageFilter.MaxFilter(3))

    image = image.resize((512, 512))
    mask = mask.resize((512, 512))

    # 6. Generate
    print("\n--- 6. Running Inpainting ---")
    with torch.autocast(device):
        result = pipe(
            prompt_embeds=target_embedding.to(dtype=pipe.unet.dtype),
            negative_prompt_embeds=sd_uncond.to(dtype=pipe.unet.dtype),
            image=image,
            mask_image=mask,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance_scale,
        ).images[0]

    print(f"Saving output to: {args.out}")
    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else ".", exist_ok=True)
    result.save(args.out)
    print("Projected Vector Injection Complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--mask", type=str, required=True)
    parser.add_argument("--out", type=str, default="zero_shot_translation/output_projected.png")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sd-model", type=str, default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--biomedclip-path", type=str, default="./saliency_maps/model")
    parser.add_argument("--projector-path", type=str, default="./zero_shot_translation/projector_medpix.pt")
    parser.add_argument("--tumor-anchor", type=str, default="tumor breast")
    parser.add_argument("--healthy-anchor", type=str, default="healthy breast")
    parser.add_argument("--dilate-mask", type=int, default=0)
    args = parser.parse_args()
    main(args)
