import argparse
import os
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor
from torchvision import transforms

from biomed_ddbm.model import BioMedDDBM

def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load BioMedCLIP
    print("Loading BioMedCLIP...")
    biomed_name = "chuhac/BiomedCLIP-vit-bert-hf"
    biomed_model = AutoModel.from_pretrained(biomed_name, trust_remote_code=True).to(device)
    biomed_processor = AutoProcessor.from_pretrained(biomed_name, trust_remote_code=True)
    biomed_model.eval()
    
    # 2. Load DDBM Model
    print(f"Loading BioMedDDBM from {args.checkpoint}...")
    model = BioMedDDBM(
        dim=64,
        biomed_img_dim=512,
        biomed_text_dim=512,
        channels=3, 
        in_channels=7 # MASK GUIDED
    ).to(device)
    
    if getattr(args, 'c_skip_boost', None) is not None:
        model.c_skip_boost = args.c_skip_boost
        print(f"Applying c_skip_boost: {model.c_skip_boost}x")
    # If checkpoint is from old training (3ch), this will fail.
    # User knows we are retraining.
    try:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Assuming you are starting fresh or checkpoint is incompatible.")
        return

    model.eval()
    
    # 3. Setup Transforms
    img_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])
    
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    
    # Denormalize for saving
    def denormalize(tensor):
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        return tensor * std + mean

    # 4. Input Handling
    print(f"Processing {args.input_image}...")
    img_pil = Image.open(args.input_image).convert("RGB")
    x_source = img_transform(img_pil).unsqueeze(0).to(device) # [1, 3, 224, 224]
    
    # Mask Loading
    if args.mask_image and os.path.exists(args.mask_image):
        mask_pil = Image.open(args.mask_image).convert("L")
    else:
        # Try inferring mask path
        base, _ = os.path.splitext(args.input_image)
        # Try common patterns
        candidates = [f"{base}_mask.png", f"{base}_mask.jpg"]
        # Try replacing 'images' with 'masks' if standard structure
        if "images" in args.input_image:
             candidates.append(args.input_image.replace("images", "masks"))
             
        mask_path = None
        for c in candidates:
            if os.path.exists(c):
                mask_path = c
                break
        
        if mask_path:
            print(f"  Found inferred mask: {mask_path}")
            mask_pil = Image.open(mask_path).convert("L")
        else:
            print("  Warning: No mask found. Using full-image mask (change everything).")
            mask_pil = Image.new("L", img_pil.size, 255) # 255 = Inpaint All

    mask = mask_transform(mask_pil).unsqueeze(0).to(device) # [1, 1, H, W]
    mask = (mask > 0.5).float() # Threshold
    
    # Masked Context (Background to PRESERVE)
    # Mask=1 -> Tumor -> Inpaint
    # Mask=0 -> Background -> Keep
    # masked_context = x_source * (1 - mask)
    masked_context = x_source * (1 - mask)
    
    # 5. Extract Embeddings (Conditions)
    with torch.no_grad():
        # Source Context
        vision_out = biomed_model.vision_model(x_source)
        c_img = vision_out[0][:, 0, :]
        if hasattr(biomed_model, 'visual_projection'):
             c_img = biomed_model.visual_projection(c_img)
        c_img = c_img / c_img.norm(dim=-1, keepdim=True)
        
        # Target Prompt
        print(f"Target Prompt: '{args.prompt}'")
        inputs = biomed_processor(text=[args.prompt], return_tensors="pt", padding=True).to(device)
        text_out = biomed_model.text_model(**inputs)
        c_text = text_out[0][:, 0, :]
        if hasattr(biomed_model, 'text_projection'):
             c_text = biomed_model.text_projection(c_text)
             
        c_text = c_text.unsqueeze(1) # [1, 1, 512]
        

        
    # 6. Sampling Loop
    num_steps = 1000
    # Schedule
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=device), alphas_cumprod[:-1]])
    
    x_t = torch.randn_like(x_source)
    
    model.eval()
    timesteps = list(range(num_steps - 1, -1, -1))
    
    # Pre-compute text features for guidance to avoid re-computing
    if args.clip_guidance_scale > 0:
        # We need the text embedding from the TEXT MODEL (not projected yet if we use raw similarity, but usually projected)
        # BioMedCLIP uses projected features for similarity.
        # c_text is already projected if model handles it, let's check.
        # In generate loop we passed c_text to model.
        # Let's re-compute a detached reference for guidance.
        with torch.no_grad():
            inputs = biomed_processor(text=[args.prompt], return_tensors="pt", padding=True).to(device)
            text_features = biomed_model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    for i, t_idx in enumerate(tqdm(timesteps)):
        t = torch.tensor([t_idx], device=device).long()
        
        # 1. Gradient Guidance (Require Grad on x_t)
        if args.clip_guidance_scale > 0:
            # We need to optimize x_t to minimize distance to text
            # But standard way is to modify pred_noise
            x_in = x_t.detach().requires_grad_(True)
            
            # Model Forward
            model_input = torch.cat([x_in, mask, masked_context], dim=1)
            pred_noise = model(model_input, t, c_img, c_text)
            
            # Check dimensions and alignment
            # pred_noise can be diff due to internal model logic, but here it should be same shape
            
            # Est x_0
            alpha_cumprod_t = alphas_cumprod[t_idx]
            pred_x0 = (x_in - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            
            # Post-process for CLIP (Denormalize from [-1,1] to [0,1], then Normalize using CLIP mean/std)
            # BioMedCLIP processor usually handles raw images.
            # We need differentiable transform.
            # 1. [-1, 1] -> [0, 1]
            pred_x0_norm = (pred_x0 + 1) / 2
            
            # 2. Resize to 224x224 (BioMedCLIP size)
            pred_x0_resized = F.interpolate(pred_x0_norm, size=(224, 224), mode='bilinear', align_corners=False)
            
            # 3. Normalize (Manual stats for BioMedCLIP/OpenAI CLIP usually)
            # BioMedCLIP uses: mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
            mu = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
            sigma = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
            clip_in = (pred_x0_resized - mu) / sigma
            
            # Get Image Features
            image_features = biomed_model.get_image_features(pixel_values=clip_in)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Similarity Loss (Negative Cosine Similarity)
            loss = -(image_features * text_features).sum()
            
            # Gradient
            grad = torch.autograd.grad(loss, x_in)[0]
            
            # Update Noise
            # score = score - sqrt(1-alpha)*grad
            # epsilon = epsilon + sqrt(1-alpha)*grad
            # We want to minimize loss -> move x_t against gradient of loss w.r.t x_t?
            # grad calculated is d(loss)/d(x_t).
            # So we shift x_t in direction -grad.
            # In terms of epsilon space: eps_new = eps + scale * grad?
            # Let's look at GLIDE: eps = eps - scale * sqrt(1-alpha) * grad(score)
            # Our "score" is -loss.
            # So eps = eps - scale * sqrt(1-alpha) * grad(-loss) = eps + scale * ... * grad(loss)
            # Confusing.
            # Simpler thinking:
            # We want x_0 to look more like text.
            # grad = d(loss)/d(x_t). Loss is high if bad.
            # We want x_t to move in -grad direction.
            # x_{t-1} is derived from x_t - eps.
            # If we modify eps: x_{t-1} ~ x_t - (eps + correction).
            # = x_t - eps - correction.
            # We want -correction to be -grad?
            # So correction = grad.
            # So eps_new = eps + scale * grad.
            
            pred_noise = pred_noise.detach() + args.clip_guidance_scale * torch.sqrt(1 - alpha_cumprod_t) * grad
            
            # detach checks
            pred_noise = pred_noise.detach()
            
        else:
            # Standard Prediction
            # Input: [x_t, mask, masked_context]
            model_input = torch.cat([x_t, mask, masked_context], dim=1)
            
            with torch.no_grad():
                pred_noise = model(model_input, t, c_img, c_text)
            
        # 2. DDPM Update (Reverse Process)
        beta_t = betas[t_idx]
        alpha_t = alphas[t_idx]
        alpha_cumprod_t = alphas_cumprod[t_idx]
        alpha_cumprod_prev_t = alphas_cumprod_prev[t_idx]
        
        # Predict x_0 from noise
        # x_0 = (x_t - sqrt(1-alpha_hat)*eps) / sqrt(alpha_hat)
        pred_x0 = (x_t - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
        pred_x0 = torch.clamp(pred_x0, -3, 3) # Clip for stability
        
        # Posterior Mean
        # mu_t = ...
        coef1 = (torch.sqrt(alpha_cumprod_prev_t) * beta_t) / (1 - alpha_cumprod_t)
        coef2 = (torch.sqrt(alpha_t) * (1 - alpha_cumprod_prev_t)) / (1 - alpha_cumprod_t)
        mu_t = coef1 * pred_x0 + coef2 * x_t
        
        # Variance
        if t_idx > 0:
            log_var_t = torch.log(torch.clamp(beta_t * (1 - alpha_cumprod_prev_t) / (1 - alpha_cumprod_t), min=1e-20))
            sigma_t = torch.exp(0.5 * log_var_t)
            noise = torch.randn_like(x_t)
            x_prev = mu_t + sigma_t * noise
        else:
            x_prev = mu_t
            
        # 3. Explicit Background Preservation (Optional but recommended)
        # Force the background to match the Forward Process of the original image
        if t_idx > 0:
            # Noise the original image to level t-1
            # q_sample(x_source, t-1)
            noise_source = torch.randn_like(x_source)
            alpha_prev = alphas_cumprod_prev[t_idx] # alpha_bar_{t-1}
            x_source_noisy = torch.sqrt(alpha_prev) * x_source + torch.sqrt(1 - alpha_prev) * noise_source
            
            # Combine: Mask -> Generated, NoMask -> Noised Source
            x_t = mask * x_prev + (1 - mask) * x_source_noisy
        else:
            x_t = mask * x_prev + (1 - mask) * x_source
            
    # 7. Final Output
    x_out = denormalize(x_t)
    x_out = torch.clamp(x_out, 0, 1)
    
    # Save Image
    x_out_np = x_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    x_out_pil = Image.fromarray((x_out_np * 255).astype(np.uint8))
    x_out_pil.save(args.output)
    print(f"Saved result at {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--mask_image", type=str, default=None, help="Optional explicit mask path")
    parser.add_argument("--prompt", type=str, default="healthy tissue")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.png")
    parser.add_argument("--clip_guidance_scale", type=float, default=0.0, help="BioMedCLIP Gradient Guidance Scale")
    parser.add_argument("--c_skip_boost", type=float, default=1.0, help="Multiply c_skip by this factor (e.g., 1.2 for +20%) to preserve original context")
    args = parser.parse_args()
    
    generate(args)
