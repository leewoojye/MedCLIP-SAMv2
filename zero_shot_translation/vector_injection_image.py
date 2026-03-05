import torch
from diffusers import StableDiffusionInpaintPipeline
from transformers import AutoTokenizer, AutoProcessor, AutoModel
import torch.nn.functional as F
import argparse
from PIL import Image

def load_biomedclip(device="cuda"):
    base_model_name = "chuhac/BiomedCLIP-vit-bert-hf"
    print(f"Loading BiomedCLIP Model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(base_model_name, trust_remote_code=True).to(device)
    
    if hasattr(model.config, "text_config") and not hasattr(model.config.text_config, "is_decoder"):
        model.config.text_config.is_decoder = False
    model.eval()
    return tokenizer, processor, model

def get_text_embedding(tokenizer, model, texts, device="cuda"):
    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    vocab_size = model.config.text_config.vocab_size
    
    if inputs["input_ids"].max() >= vocab_size:
        inputs["input_ids"] = torch.clamp(inputs["input_ids"], max=vocab_size - 1)
        
    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    seq_length = inputs["input_ids"].shape[1]
    batch_size = inputs["input_ids"].shape[0]
    inputs["position_ids"] = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)
    inputs = inputs.to(device)
    
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

def get_image_embedding(processor, model, image_paths, device="cuda"):
    images = [Image.open(p).convert("RGB") for p in image_paths]
    inputs = processor(images=images, return_tensors="pt").to(device)
    
    with torch.no_grad():
        vision_outputs = model.vision_model(
            pixel_values=inputs["pixel_values"],
            output_hidden_states=True
        )
        # return unprojected hidden states: (batch_size, seq_len=197, hidden_size=768)
        sequence_output = vision_outputs[0]
        
    return sequence_output

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("\n--- 1. Computing Semantic Vector Shift from Image Pair ---")
    bio_tokenizer, bio_processor, bio_model = load_biomedclip(device=device)
    
    print(f"Using Image Anchors:\n Tumor (Source): {args.img_tumor}\n Healthy (Source): {args.img_healthy}")
    embs_source = get_image_embedding(bio_processor, bio_model, [args.img_tumor, args.img_healthy], device=device)
    
    v_tumor_img = embs_source[0:1] # (1, 197, 768)
    v_healthy_img = embs_source[1:2] # (1, 197, 768)
    
    # Delta V = Healthy Image - Tumor Image (Pixel-wise aligned shift in ViT latent space)
    shift_vector = v_healthy_img - v_tumor_img # (1, 197, 768)
    
    print(f"\n--- 2. Computing Base Target Condition ---")
    if args.use_text_target:
        # Base condition is a text prompt (e.g. "tumor breast")
        print(f"Target Condition: TEXT ['{args.target_prompt}']")
        target_base = get_text_embedding(bio_tokenizer, bio_model, [args.target_prompt], device=device)
        
        # We align the 197-token image shift to the text sequence length using CLS token
        cls_shift = shift_vector[:, 0:1, :] 
        shift_vector_aligned = cls_shift.expand(-1, target_base.shape[1], -1)
    else:
        # Base condition is the input image itself
        print(f"Target Condition: IMAGE ['{args.image}']")
        target_base = get_image_embedding(bio_processor, bio_model, [args.image], device=device)
        # Direct sequence addition since both are ViT outputs (197 tokens)
        shift_vector_aligned = shift_vector 

    alpha = args.alpha
    print(f"Injecting shift vector with alpha = {alpha}")
    target_embedding = target_base + (alpha * shift_vector_aligned)
    
    # Unconditional negative embedding
    uncond_emb = get_text_embedding(bio_tokenizer, bio_model, [""], device=device)
    if uncond_emb.shape[1] < target_embedding.shape[1]:
        pad_size = target_embedding.shape[1] - uncond_emb.shape[1]
        uncond_emb = F.pad(uncond_emb, (0, 0, 0, pad_size), "constant", 0)
    elif uncond_emb.shape[1] > target_embedding.shape[1]:
        uncond_emb = uncond_emb[:, :target_embedding.shape[1], :]

    print(f"Target Embedding Shape (Input to SD): {target_embedding.shape}")

    # 3. Setup SD Inpainting Pipeline
    print(f"\n--- 3. Loading Stable Diffusion Inpainting Model ---")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.sd_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    pipe.safety_checker = None
    
    # 4. Load Target Images
    print(f"\n--- 4. Processing Target Image & Mask ---")
    image = Image.open(args.image).convert("RGB").resize((512, 512))
    mask = Image.open(args.mask).convert("L").resize((512, 512))
    
    # 5. Generate Image
    print("\n--- 5. Running Generation ---")
    with torch.autocast(device):
        result = pipe(
            prompt_embeds=target_embedding.to(dtype=pipe.unet.dtype),
            negative_prompt_embeds=uncond_emb.to(dtype=pipe.unet.dtype),
            image=image,
            mask_image=mask,
            num_inference_steps=50,
            guidance_scale=7.5,
        ).images[0]
        
    print(f"Saving output to: {args.out}")
    result.save(args.out)
    print("Zero-Shot Translation Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_tumor", type=str, required=True, help="Path to original source tumor image (anchor)")
    parser.add_argument("--img_healthy", type=str, required=True, help="Path to negative matched healthy image (anchor)")
    parser.add_argument("--image", type=str, required=True, help="Path to target image to inpaint")
    parser.add_argument("--mask", type=str, required=True, help="Path to mask for target image")
    parser.add_argument("--out", type=str, default="zero_shot_translation/output_img_injection.png")
    parser.add_argument("--alpha", type=float, default=1.0, help="Intensity of the healing shift")
    parser.add_argument("--sd_model", type=str, default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--use_text_target", action="store_true", help="Use text as the base condition instead of target image")
    parser.add_argument("--target_prompt", type=str, default="tumor breast", help="Text condition if using --use_text_target")
    args = parser.parse_args()
    main(args)
