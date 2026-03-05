import torch
from diffusers import StableDiffusionInpaintPipeline
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import argparse
from PIL import Image, ImageFilter

def load_biomedclip(device="cuda"):
    base_model_name = "chuhac/BiomedCLIP-vit-bert-hf"
    print(f"Loading BiomedCLIP Model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True).to(device)
    
    if not hasattr(model.config.text_config, "is_decoder"):
        model.config.text_config.is_decoder = False
    model.eval()
    return tokenizer, model

def get_text_embedding(tokenizer, model, texts, device="cuda"):
    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    vocab_size = model.config.text_config.vocab_size
    
    # Clamp input_ids to prevent out-of-bounds error
    if inputs["input_ids"].max() >= vocab_size:
        inputs["input_ids"] = torch.clamp(inputs["input_ids"], max=vocab_size - 1)
        
    # Bypass model's corrupted internal buffer by explicitly zeroing it out
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    
    # Bypass position_ids corrupted buffer
    seq_length = inputs["input_ids"].shape[1]
    batch_size = inputs["input_ids"].shape[0]
    inputs["position_ids"] = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)
    
    inputs = inputs.to(device)
    
    with torch.no_grad():
        features = model.get_text_features(**inputs)
        # We need the raw embedding layer outputs or the projected text features.
        # SD 1.5 expects (batch_size, sequence_length, hidden_size) for prompt_embeds.
        # However, BioMedCLIP get_text_features returns the projected pooled vector (batch_size, projection_dim).
        # We need the unprojected sequence embeddings to inject into SD!
        
        # Let's get the raw hidden states from the text encoder:
        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            output_hidden_states=True
        )
        # text_outputs[0] is the sequence output of shape (batch_size, seq_len, hidden_size)
        sequence_output = text_outputs[0]
        
    return sequence_output

def get_image_embedding(processor, model, image_paths, device="cuda"):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    
    with torch.no_grad():
        features = model.get_image_features(**inputs)
        # To align with SD, expand to sequence length 4 like we did for text
        features = features.unsqueeze(1).expand(-1, 4, -1)
    return features

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Calculate Shift Vector from BioMedCLIP Text Anchors
    print("\n--- 1. Computing Semantic Vector Shift from Text ---")
    bio_tokenizer, bio_model = load_biomedclip(device=device)
    
    # Use text anchors to create the shift
    # MODIFIED: For breast translation, we just use the breast anchors directly.
    print(f"Using Text Anchors: '{args.healthy_anchor}' - '{args.tumor_anchor}'")
    phrases = [args.tumor_anchor, args.healthy_anchor]
    embs = get_text_embedding(bio_tokenizer, bio_model, phrases, device=device)
    
    v_tumor_anchor = embs[0:1] # (1, seq_len, hidden_size)
    v_healthy_anchor = embs[1:2]
    
    # Delta V = Healthy - Tumor (The "Healing" Shift)
    shift_vector = v_healthy_anchor - v_tumor_anchor
    
    # New Condition = Original Target Anchor (Tumor Breast) + Alpha * Shift
    # Here the target anchor is just the tumor anchor itself since we are doing breast -> breast.
    alpha = args.alpha
    print(f"Injecting shift vector with alpha = {alpha}")
    target_embedding = v_tumor_anchor + (alpha * shift_vector)
    
    # Unconditional negative embedding
    uncond_emb = get_text_embedding(bio_tokenizer, bio_model, [""], device=device)
    
    # Expand unconditional to match sequence length if necessary, or pad.
    if uncond_emb.shape[1] < target_embedding.shape[1]:
        pad_size = target_embedding.shape[1] - uncond_emb.shape[1]
        uncond_emb = F.pad(uncond_emb, (0,0, 0, pad_size), "constant", 0)
    elif uncond_emb.shape[1] > target_embedding.shape[1]:
        uncond_emb = uncond_emb[:, :target_embedding.shape[1], :]

    print(f"Target Embedding Shape (Input to SD): {target_embedding.shape}")

    # 2. Setup SD Inpainting Pipeline
    print(f"\n--- 2. Loading Stable Diffusion Inpainting Model ---")
    print(f"Model ID: {args.sd_model}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.sd_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    
    # Disable safety checker for medical images
    pipe.safety_checker = None
    
    # 3. Load Images
    print(f"\n--- 3. Processing Image ---")
    image = Image.open(args.image).convert("RGB")
    mask = Image.open(args.mask).convert("L")  # Grayscale mask (white = inpaint area)
    
    if args.dilate_mask > 0:
        print(f"Dilating mask by {args.dilate_mask} iterations...")
        for _ in range(args.dilate_mask):
            mask = mask.filter(ImageFilter.MaxFilter(3))
            
    # Resize to standard SD size (e.g., 512x512)
    image = image.resize((512, 512))
    mask = mask.resize((512, 512))
    
    # 4. Generate Image (Zero-Shot Translation)
    print("\n--- 4. Running Generation ---")
    # Normally we pass `prompt`, but here we pass `prompt_embeds` directly!
    with torch.autocast(device):
        result = pipe(
            prompt_embeds=target_embedding.to(dtype=pipe.unet.dtype),
            negative_prompt_embeds=uncond_emb.to(dtype=pipe.unet.dtype),
            image=image,
            mask_image=mask,
            num_inference_steps=50,
            guidance_scale=5.0,
        ).images[0]
        
    print(f"Saving output to: {args.out}")
    result.save(args.out)
    print("Zero-Shot Translation Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to original medical image")
    parser.add_argument("--mask", type=str, required=True, help="Path to mask (white=tumor area to paint over)")
    parser.add_argument("--out", type=str, default="zero_shot_translation/output_breast.png")
    parser.add_argument("--alpha", type=float, default=1.0, help="Intensity of the healing shift vector")
    parser.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-inpainting")
    # Added specific anchor arguments to make it easily customizable
    parser.add_argument("--tumor_anchor", type=str, default="tumor breast", help="The starting semantic text (e.g., tumor breast)")
    parser.add_argument("--healthy_anchor", type=str, default="healthy breast", help="The target semantic text (e.g., healthy breast)")
    parser.add_argument("--dilate_mask", type=int, default=0, help="Number of iterations to dilate the mask (0 to disable)")
    args = parser.parse_args()
    main(args)
