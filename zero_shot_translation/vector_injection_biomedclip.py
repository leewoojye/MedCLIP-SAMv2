import torch
from diffusers import StableDiffusionInpaintPipeline
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import argparse
from PIL import Image, ImageFilter
import os

def load_biomedclip_and_sd(sd_model_id, biomedclip_path, ckpt_path, device="cuda"):
    # 1. Load BioMedCLIP (DHN)
    print(f"Loading BiomedCLIP from: {biomedclip_path}")
    bio_tokenizer = AutoTokenizer.from_pretrained(
        "chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True
    )
    bio_model = AutoModel.from_pretrained(
        biomedclip_path, trust_remote_code=True
    ).to(device)
    
    if not hasattr(bio_model.config.text_config, "is_decoder"):
        bio_model.config.text_config.is_decoder = False
    bio_model.eval()

    # 2. Load SD Inpainting Pipeline
    print(f"Loading SD Inpainting Pipeline: {sd_model_id}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        sd_model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    pipe.safety_checker = None

    # 3. Inject Trained Cross-Attention Weights
    print(f"Injecting trained Cross-Attention weights from: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    
    # Cast weights to pipeline dtype (e.g., float16)
    target_dtype = pipe.unet.dtype
    state_dict = {k: v.to(target_dtype) for k, v in state_dict.items()}
    
    # Check what keys are in state_dict
    missing_keys, unexpected_keys = pipe.unet.load_state_dict(state_dict, strict=False)
    print(f"Loaded {len(state_dict)} tensors into U-Net.")
    
    return bio_tokenizer, bio_model, pipe

def get_text_embedding(tokenizer, model, texts, device="cuda"):
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
            output_hidden_states=True
        )
        sequence_output = text_outputs[0]
        
    return sequence_output

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    bio_tokenizer, bio_model, pipe = load_biomedclip_and_sd(
        sd_model_id=args.sd_model,
        biomedclip_path=args.biomedclip_path,
        ckpt_path=args.ckpt_path,
        device=device
    )
    
    # --- 1. Compute Semantic Vector ---
    # In this approach, we just use the target anchor (e.g., "healthy breast") directly,
    # because the U-Net has learned to understand BiomedCLIP embeddings natively.
    # Optionally, we can still use the shift vector logic. Let's provide the target directly first.
    
    print("\n--- 1. Computing Text Embeddings ---")
    if args.alpha != 0.0:
        # Shift vector approach
        print(f"Using Shift Logic: '{args.healthy_anchor}' - '{args.tumor_anchor}' with alpha={args.alpha}")
        embs = get_text_embedding(bio_tokenizer, bio_model, [args.tumor_anchor, args.healthy_anchor], device=device)
        v_tumor = embs[0:1]
        v_healthy = embs[1:2]
        shift = v_healthy - v_tumor
        target_embedding = v_tumor + (args.alpha * shift)
    else:
        # Direct approach
        print(f"Using Direct Target Anchor: '{args.healthy_anchor}'")
        target_embedding = get_text_embedding(bio_tokenizer, bio_model, [args.healthy_anchor], device=device)

    # Convert to pipeline dtype
    target_embedding = target_embedding.to(dtype=pipe.unet.dtype)

    # Unconditional negative embedding
    uncond_emb = get_text_embedding(bio_tokenizer, bio_model, [""], device=device).to(dtype=pipe.unet.dtype)
    
    if uncond_emb.shape[1] < target_embedding.shape[1]:
        pad_size = target_embedding.shape[1] - uncond_emb.shape[1]
        uncond_emb = F.pad(uncond_emb, (0,0, 0, pad_size), "constant", 0)
    elif uncond_emb.shape[1] > target_embedding.shape[1]:
        uncond_emb = uncond_emb[:, :target_embedding.shape[1], :]

    print(f"Target Embedding Shape (Input to SD): {target_embedding.shape}")

    # --- 2. Process Image ---
    print("\n--- 2. Processing Image ---")
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
    
    # --- 3. Generate Image ---
    print("\n--- 3. Running Generation ---")
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
    # Resize back to original
    result = result.resize(orig_size, Image.LANCZOS)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    result.save(args.out)
    print("Generation Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to original medical image")
    parser.add_argument("--mask", type=str, required=True, help="Path to mask")
    parser.add_argument("--out", type=str, default="zero_shot_translation/output_biomedclip.png")
    parser.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--biomedclip_path", type=str, default="./saliency_maps/model")
    parser.add_argument("--ckpt_path", type=str, default="./zero_shot_translation/sd_biomedclip_ckpt/best.pt")
    
    # We can use direct prompting (alpha=0) or shift prompting (alpha > 0)
    parser.add_argument("--alpha", type=float, default=0.0, help="0: Direct prompt. >0: Shift vector intensity")
    parser.add_argument("--tumor_anchor", type=str, default="breast ultrasound with tumor")
    parser.add_argument("--healthy_anchor", type=str, default="healthy breast ultrasound, normal tissue")
    
    parser.add_argument("--dilate_mask", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    args = parser.parse_args()
    main(args)
