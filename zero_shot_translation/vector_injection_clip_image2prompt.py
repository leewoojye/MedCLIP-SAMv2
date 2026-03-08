import torch
import torch.nn as nn
from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPVisionModel, CLIPImageProcessor
import torch.nn.functional as F
import argparse
from PIL import Image, ImageFilter
import os

class ImageToPromptProjector(nn.Module):
    """Linear projection from CLIP Vision (1024) to SD Text (768)."""
    def __init__(self, in_dim=1024, out_dim=768):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        return self.proj(x)

def load_clip_vision(device="cuda"):
    print("Loading CLIP Vision Model (openai/clip-vit-large-patch14)...")
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
    processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()
    return processor, model

def get_image_embedding(processor, model, image_path, device="cuda"):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # unprojected hidden states: (batch_size, 257, 1024)
        hidden_states = outputs.hidden_states[-1] 
    return hidden_states

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 1. Load Vision Model & Compute Image-as-Prompt Embedding ---
    print("\n--- 1. Computing Image-as-Prompt Vector ---")
    clip_processor, clip_vision = load_clip_vision(device=device)
    
    print(f"Extracting visual features from normal image: {args.normal_image}")
    # target_embedding_1024 shape: [1, 257, 1024]
    target_embedding_1024 = get_image_embedding(clip_processor, clip_vision, args.normal_image, device=device)
    print(f"Visual prompt embedding shape: {target_embedding_1024.shape}")

    # --- 2. Project Image Embedding to Text Embedding Space ---
    print("\n--- 2. Projecting Image to Text Embedding Space (1024 -> 768) ---")
    
    # We load the raw derived projection matrix W_map of shape (768, 1024)
    # y = x * W^T + b. For our projection y = x * W_map^T, nn.Linear(1024, 768, bias=False)
    projector = nn.Linear(1024, 768, bias=False).to(device)
    
    if args.projector_path and os.path.exists(args.projector_path):
        print(f"Loading derived closed-form projector from {args.projector_path}")
        # The file is just a tensor: torch.Size([768, 1024])
        # Linear layer weight expects shape (out_features, in_features) -> (768, 1024)
        W_map = torch.load(args.projector_path, map_location=device)
        if hasattr(projector, "weight"):
            projector.weight.data = W_map
    else:
        print("Note: No trained projector provided. Using random initialization linear layer for pipeline testing.")
    
    projector.eval()
    with torch.no_grad():
        target_embedding = projector(target_embedding_1024) # [1, 257, 768]
        
    print(f"Projected prompt embedding shape: {target_embedding.shape}")
    print(f"Projected prompt norm before scaling: {target_embedding.norm(dim=-1).mean().item():.2f}")
    
    # --- OPTIONAL: Scale Matching ---
    # The derived W_map amplified the scale significantly (norm ~4000 vs expected ~30 for SD text).
    # We explicitly scale the embedded tensor to match appropriate text feature norms.
    target_norm = target_embedding.norm(dim=-1, keepdim=True)
    # Target scale roughly matches typical Stable Diffusion text tokens.
    expected_norm = 30.0 
    target_embedding = target_embedding * (expected_norm / target_norm)
    print(f"Projected prompt norm after scaling:  {target_embedding.norm(dim=-1).mean().item():.2f}")

    # --- 3. Load SD Inpainting Pipeline ---
    print(f"\n--- 3. Loading Stable Diffusion Inpainting Model ---")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.sd_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    pipe.safety_checker = None

    target_embedding = target_embedding.to(dtype=pipe.unet.dtype)

    # Calculate unconditional text embedding ("", sequence length 77, hidden size 768)
    print("\n--- 4. Computing Negative Prompt Embedding ---")
    prompt_embeds_len = target_embedding.shape[1]
    text_inputs = pipe.tokenizer(
        "",
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_emb = pipe.text_encoder(text_inputs.input_ids.to(device))[0] # [1, 77, 768]
    uncond_emb = uncond_emb.to(dtype=pipe.unet.dtype)
    
    # Pad from 77 to 257
    if uncond_emb.shape[1] < prompt_embeds_len:
        pad_size = prompt_embeds_len - uncond_emb.shape[1]
        uncond_emb = F.pad(uncond_emb, (0, 0, 0, pad_size), "constant", 0)
    
    # --- 5. Process Target Image & Mask ---
    print(f"\n--- 5. Processing Target Image & Mask ---")
    image = Image.open(args.image).convert("RGB")
    mask = Image.open(args.mask).convert("L")
    
    if args.dilate_mask > 0:
        print(f"Dilating mask by {args.dilate_mask} iterations...")
        for _ in range(args.dilate_mask):
            mask = mask.filter(ImageFilter.MaxFilter(3))
            
    # Keep original size for output resizing later
    orig_size = image.size
    image = image.resize((512, 512), Image.LANCZOS)
    mask = mask.resize((512, 512), Image.NEAREST)
    
    # --- 6. Generate Image ---
    print("\n--- 6. Running Generation ---")
    with torch.autocast(device):
        result = pipe(
            prompt_embeds=target_embedding,
            negative_prompt_embeds=uncond_emb,
            image=image,
            mask_image=mask,
            num_inference_steps=50,
            guidance_scale=args.guidance_scale,
        ).images[0]
        
    print(f"Saving output to: {args.out}")
    result = result.resize(orig_size, Image.LANCZOS)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result.save(args.out)
    print("Zero-Shot Image-to-Prompt Injection Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to target image to inpaint")
    parser.add_argument("--mask", type=str, required=True, help="Path to mask for target image")
    parser.add_argument("--normal_image", type=str, required=True, help="Path to normal ultrasound image to use as prompt mapping")
    parser.add_argument("--out", type=str, default="zero_shot_translation/output_image2prompt.png")
    parser.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--projector_path", type=str, default="", help="Path to trained 1024->768 projector")
    parser.add_argument("--dilate_mask", type=int, default=3)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    args = parser.parse_args()
    main(args)
