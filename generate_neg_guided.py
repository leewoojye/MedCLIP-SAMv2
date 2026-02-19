print("STARTING SCRIPT...", flush=True)
import torch
from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
from PIL import Image
import numpy as np
import os
from transformers import AutoProcessor, AutoModel
from tqdm import tqdm

# 1. Load BiomedCLIP (for guidance)
print("Loading BiomedCLIP...")
biomed_model_name = "chuhac/BiomedCLIP-vit-bert-hf"
biomed_model = AutoModel.from_pretrained(biomed_model_name, trust_remote_code=True).to("cuda")
biomed_processor = AutoProcessor.from_pretrained(biomed_model_name, trust_remote_code=True)
biomed_model.eval()

# 2. Load SD Inpainting Pipeline
print("Loading Stable Diffusion Inpainting...")
model_id = "stable-diffusion-v1-5/stable-diffusion-inpainting"
# Use DDIM for consistent inversion/stepping which is often better for guidance
scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    model_id,
    scheduler=scheduler,
    torch_dtype=torch.float16,
).to("cuda")

# LoRA Loading (Original was likely breast-specific, disabling for generalizability)
# lora_model_dir = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/finetune4neg/finetuned_output"
# try:
#     pipe.load_lora_weights(lora_model_dir, weight_name="pytorch_lora_weights.safetensors")
#     print(f"Loaded LoRA weights from {lora_model_dir}")
# except Exception as e:
#     print(f"Warning: Could not load LoRA weights: {e}")

# Helper for resizing (Same as original)
def resize_and_pad_square(img: Image.Image, size: int, resample: int, mode: str = "RGB") -> Image.Image:
    img = img.convert(mode)
    w, h = img.size
    scale = size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    img_resized = img.resize((new_w, new_h), resample=resample)
    canvas = Image.new(mode, (size, size))
    offset = ((size - new_w) // 2, (size - new_h) // 2)
    canvas.paste(img_resized, offset)
    return canvas

# Guidance Logic
def generate_healthy_guided(image_path, mask_path, save_path, guidance_target_text="healthy tissue", guidance_scale=100.0):
    # Setup
    target_size = int(os.getenv("INPAINT_SIZE", "768"))
    init_image = resize_and_pad_square(Image.open(image_path), target_size, Image.BICUBIC, mode="RGB")
    mask_image = resize_and_pad_square(Image.open(mask_path), target_size, Image.NEAREST, mode="RGB")
    
    # Preprocess for SD
    # SD expects range [-1, 1] for image and [0, 1] for mask usually handled by pipeline, but we need manual control
    # Actually checking pipeline source, prepare_mask_and_masked_image handles this.
    # We will use the pipeline's components to prepare latents.
    
    prompt = "healthy brain MRI, homogeneous brain tissue, T1 weighted, medical imaging, high quality"
    negative_prompt = "tumor, mass, lesion, cyst, edema, abnormal growth, cancer, glioma, meningioma, pituitary"
    num_inference_steps = 70 # As per original

    # Encode prompts
    with torch.no_grad():
        prompt_embeds = pipe._encode_prompt(
            prompt,
            "cuda",
            1, # num_images_per_prompt
            True, # do_classifier_free_guidance
            negative_prompt,
        )

    # Manual prepare mask and masked_image
    # Convert init_image (PIL) to tensor [-1, 1]
    image_tensor = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
    image_tensor = image_tensor[None].transpose(0, 3, 1, 2)
    image_tensor = torch.from_numpy(image_tensor)
    image_tensor = 2.0 * image_tensor - 1.0
    image_tensor = image_tensor.to(device="cuda", dtype=torch.float16)

    # Convert mask_image (PIL) to tensor [0, 1]
    mask_tensor = np.array(mask_image.convert("L")).astype(np.float32) / 255.0
    mask_tensor = mask_tensor[None, None]
    mask_tensor[mask_tensor < 0.5] = 0
    mask_tensor[mask_tensor >= 0.5] = 1
    mask_tensor = torch.from_numpy(mask_tensor).to(device="cuda", dtype=torch.float16)

    # Compute masked_image (latents input)
    # Usually masked_image = image * (1-mask) if mask=1 is "repaint".
    masked_image_tensor = image_tensor * (mask_tensor < 0.5)

    mask = mask_tensor
    masked_image = masked_image_tensor

    # Encode masked image to latents
    with torch.no_grad():
        masked_image_latents = pipe.vae.encode(masked_image.to(dtype=torch.float16)).latent_dist.sample()
        masked_image_latents = masked_image_latents * pipe.vae.config.scaling_factor
    
    # Resize mask to latent shape
    mask = torch.nn.functional.interpolate(mask, size=(target_size // 8, target_size // 8))
    
    # NOTE: SD Inpainting model (9 channels) takes concatenation of (latents, mask, masked_image_latents)
    # The mask should be 1 channel.
    
    # Prepare initial random noise
    latents = torch.randn(
        (1, 4, target_size // 8, target_size // 8),
        device="cuda",
        dtype=torch.float16
    )
    
    pipe.scheduler.set_timesteps(num_inference_steps, device="cuda")
    timesteps = pipe.scheduler.timesteps

    # Prepare BiomedCLIP text embedding
    with torch.no_grad():
        target_inputs = biomed_processor(text=[guidance_target_text], return_tensors="pt", padding=True).to("cuda")
        text_outputs = biomed_model.text_model(**target_inputs, return_dict=True)
        # Handle tuple output if return_dict is not respected or defaults to tuple in some versions
        if hasattr(text_outputs, 'last_hidden_state'):
            target_text_embeds = text_outputs.last_hidden_state[:, 0, :]
        else:
            target_text_embeds = text_outputs[0][:, 0, :]
        target_text_embeds = target_text_embeds / target_text_embeds.norm(dim=-1, keepdim=True)

    # Denoising Loop
    for i, t in enumerate(tqdm(timesteps)):
        # 1. Expand latents for CFG
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        # 2. Concatenate with mask and masked_image_latents for inpainting model
        # We need to repeat mask and masked_image for CFG
        mask_input = torch.cat([mask] * 2)
        masked_image_latents_input = torch.cat([masked_image_latents] * 2)
        
        latent_model_input = torch.cat([latent_model_input, mask_input, masked_image_latents_input], dim=1)

        # 3. Predict noise
        with torch.no_grad(): # Standard UNet pass
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                return_dict=False,
            )[0]

        # CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + 9.0 * (noise_pred_text - noise_pred_uncond) # guidance_scale=9 from original

        # 4. Compute 'original sample' (x0) approximation for guidance
        # We need gradients here!
        # Re-enabling grad for this small part implies re-running UNet or using the scheduler step which is differentiable
        # BUT standard scheduler step is differentiable if tensors require grad.
        # The latents at start of loop don't require grad.
        
        # TRICK: We apply guidance to the *latents* based on the *decoded* image.
        # We need to enable grad on latents -> UNet -> Decode?
        # NO. We can apply guidance based on `latents` directly if we approximate `x0` from `latents` and `noise_pred`
        # `x0` = (latents - sqrt(1-alpha)*noise) / sqrt(alpha)
        # We can do this calculation to get x0, decode it, and compute loss.
        # However, `noise_pred` was computed without grad. 
        # If we treat `noise_pred` as constant for the gradient step, we can compute `x0` from `latents` (which we enable grad on)
        # BUT `latents` themselves are the variable we want to optimize using the gradient.
        # Correct approach for "Blue Loss" / Guidance:
        #   latents = latents.detach().requires_grad_()
        #   x0_hat = approx_x0(latents, noise_pred_fixed)
        #   image = decode(x0_hat)
        #   loss = biomed(image, text)
        #   grad = graid(loss, latents)
        #   latents = latents - grad * scale
        
        # Step 4a: Enable gradient on latents
        latents = latents.detach().requires_grad_(True)
        
        # Step 4b: Approximate x0 (Predict Original Sample)
        # DDIM/PNDM schedulers have logic for this.
        # alpha_prod_t = scheduler.alphas_cumprod[t]
        # beta_prod_t = 1 - alpha_prod_t
        # pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)
        # Using scheduler's built-in step if possible, but step moves to prev_sample.
        # Let's manually compute x0 for guidance purpose.
        
        alpha_prod_t = pipe.scheduler.alphas_cumprod[t]
        beta_prod_t = 1 - alpha_prod_t
        
        # Careful with shapes and devices
        pred_original_sample = (latents - (beta_prod_t ** 0.5) * noise_pred) / (alpha_prod_t ** 0.5)
        
        # Decode x0
        # VAE decoding is heavy. To speed up, maybe scale down? But BiomedCLIP needs 224x224.
        # VAE decoding expects scaled latents
        pred_original_sample_scaled = 1 / pipe.vae.config.scaling_factor * pred_original_sample
        
        # VAE decode (requires grad!)
        # VAE is usually float32 or float16.
        image_pred = pipe.vae.decode(pred_original_sample_scaled, return_dict=False)[0]
        
        # Process image for BiomedCLIP (resize/normalize)
        # image_pred is [-1, 1]. BiomedCLIP expects standard normalization.
        # Convert [-1, 1] to [0, 1]
        image_pred = (image_pred / 2 + 0.5).clamp(0, 1)
        
        # Resize to 224x224 for BiomedCLIP
        image_pred_resized = torch.nn.functional.interpolate(image_pred, size=(224, 224), mode='bicubic', align_corners=False)
        
        # Normalize (standard ImageNet mean/std used by CLIP usually)
        # BiomedCLIP uses: mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device="cuda").view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device="cuda").view(1, 3, 1, 1)
        image_pred_norm = (image_pred_resized - mean) / std
        
        # Compute Embeddings
        # BiomedCLIP vision model
        # BiomedCLIP vision model is a ViT.
        
        vision_outputs = biomed_model.vision_model(image_pred_norm, return_dict=True)
        if hasattr(vision_outputs, 'last_hidden_state'):
            image_embeds = vision_outputs.last_hidden_state[:, 0, :]
        else:
            image_embeds = vision_outputs[0][:, 0, :]
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        
        # Compute Loss (Cosine Distance) based on masked region or whole image?
        # User wants masked region to be guided.
        # But image_pred is the Whole Image (inpainted).
        # Guiding the whole image to be "healthy" is correct.
        
        loss = -(image_embeds * target_text_embeds).sum()
        
        # Compute Gradient
        grad = torch.autograd.grad(loss, latents)[0]
        
        # Apply Guidance
        # We subtract gradient to minimize loss (negative similarity -> maximize similarity)
        # Wait, if loss is -sim, minimizing loss maximizes sim.
        # So latents = latents - grad * scale
        
        # Optimization: Normalize gradient?
        if grad.abs().max() > 0:
             grad = grad / grad.abs().mean()
        
        latents = latents - grad * guidance_scale
        
        # Cleanup
        latents = latents.detach()
        del image_pred, loss, grad
        
        # 5. Scheduler Step (Standard)
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

    # Final Decode
    # Manually decode latents
    latents = 1 / pipe.vae.config.scaling_factor * latents
    image = pipe.vae.decode(latents, return_dict=False)[0]
    image = (image / 2 + 0.5).clamp(0, 1)
    # Convert to PIL
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    image = pipe.numpy_to_pil(image)[0]
    
    # Save
    if os.path.isdir(save_path):
        filename = os.path.basename(image_path)
        save_path = os.path.join(save_path, filename)
    image.save(save_path)
    print(f"Guided Result saved at: {save_path}")

# Example Usage Wrapper
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to image or directory")
    parser.add_argument("--mask", required=True, help="Path to mask or directory (filenames must match)")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--target", default="healthy brain MRI, homogeneous brain tissue")
    parser.add_argument("--scale", type=float, default=200.0) # Start with high scale
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images")
    args = parser.parse_args()
    
    # Check if input is directory
    if os.path.isdir(args.image):
        if not os.path.exists(args.out):
            os.makedirs(args.out)
            
        images = sorted([f for f in os.listdir(args.image) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if args.limit:
            images = images[:args.limit]
            
        print(f"Processing {len(images)} images from {args.image}...")
        
        for img_name in tqdm(images):
            img_path = os.path.join(args.image, img_name)
            mask_path = os.path.join(args.mask, img_name)
            
            if not os.path.exists(mask_path):
                print(f"Skipping {img_name}: Mask not found.")
                continue
                
            try:
                generate_healthy_guided(img_path, mask_path, args.out, args.target, args.scale)
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                import traceback
                traceback.print_exc()
                
    else:
        # Single file
        generate_healthy_guided(args.image, args.mask, args.out, args.target, args.scale)
