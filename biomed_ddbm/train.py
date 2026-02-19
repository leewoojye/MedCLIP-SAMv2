import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from biomed_dataset import EgoBridgeDataset, PairedBioMedDataset
from biomed_ddbm.model import BioMedDDBM

def train(args):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Setup BioMedCLIP (Frozen)
    print("Loading BioMedCLIP...")
    biomed_name = "chuhac/BiomedCLIP-vit-bert-hf"
    biomed_model = AutoModel.from_pretrained(biomed_name, trust_remote_code=True).to(device)
    biomed_processor = AutoProcessor.from_pretrained(biomed_name, trust_remote_code=True)
    biomed_model.eval()
    for param in biomed_model.parameters():
        param.requires_grad = False
        
    # 3. Setup Dataset
    # Paired Training Mode with Masks
    print(f"Loading Paired Dataset...")
    dataset = PairedBioMedDataset(
        pos_dir=args.pos_dir, 
        neg_dir=args.neg_dir,
        mask_dir=args.mask_dir,
        transform=transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # 4. Setup DDBM Model (Inpainting)
    print("Initializing BioMedDDBM (Inpainting Mode)...")
    # Inpainting: 3 (Noisy) + 1 (Mask) + 3 (Masked Image) = 7 Channels
    model = BioMedDDBM(
        dim=64,
        biomed_img_dim=512,
        biomed_text_dim=512,
        channels=3, # Output: Predicted Noise (3 RGB)
        in_channels=7 # Input: Noisy(3) + Mask(1) + Context(3)
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Scale Prediction instead of Noise? DDBM paper suggests predicting scores.
    # But usually standard DDPM predicting noise (epsilon) is robust.
    # We will predict x_0 or epsilon. Let's predict epsilon (noise).
    
    # 5. Training Loop
    print("Starting Training...")
    global_step = 0
    
    for epoch in range(args.epochs):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in progress_bar:
            # Batch: {'image_pos', 'image_neg', 'mask', 'type'}
            img_pos = batch['image_pos'].to(device) # Source (Tumor)
            img_neg = batch['image_neg'].to(device) # Target (Healthy)
            mask = batch['mask'].to(device) # [B, 1, H, W] (1=Tumor, 0=BG)
            
            bs = img_pos.shape[0]
            
            # Masked Context (Background of Positive Image)
            # We want to fill the mask region.
            masked_context = img_pos * (1 - mask)
            
            # A. Prepare Conditions (BioMedCLIP)
            with torch.no_grad():
                # c_img: Embedding of Positive Image (Global Context)
                # Or should it be masked context? BioMedCLIP is strong, global context helps.
                vision_out = biomed_model.vision_model(img_pos)
                c_img = vision_out[0][:, 0, :] # CLS token
                
                # Projection Fix
                if hasattr(biomed_model, 'visual_projection'):
                    c_img = biomed_model.visual_projection(c_img)
                c_img = c_img / c_img.norm(dim=-1, keepdim=True)
                
                # c_text: Embedding of "Healthy" concept (Target Context)
                prompts = ["healthy breast tissue", "normal breast mammogram", "no tumor"]
                # Start with simple fixed prompt
                text_input = [prompts[0]] * bs
                inputs = biomed_processor(text=text_input, return_tensors="pt", padding=True).to(device)
                text_out = biomed_model.text_model(**inputs)
                c_text = text_out[0][:, 0, :]
                
                # Projection Fix
                if hasattr(biomed_model, 'text_projection'):
                    c_text = biomed_model.text_projection(c_text)
                
                c_text_cond = c_text.unsqueeze(1) # [B, 1, 512] for CrossAttn
            
            # B. Noise Injection (Forward Process)
            # We perform diffusion on the 'Target' (Healthy Image)
            # Input to model: Noisy Target + Mask + Context(Pos Background)
            # We want to denoise Target, conditioned on Context.
            
            t = torch.randint(0, 1000, (bs,), device=device).long()
            
            # Simple Linear Schedule
            beta_start = 0.0001
            beta_end = 0.02
            num_steps = 1000
            betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
            alphas = 1. - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            
            noise = torch.randn_like(img_neg)
            alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1)
            
            # Add noise to target
            x_noisy = torch.sqrt(alpha_t) * img_neg + torch.sqrt(1 - alpha_t) * noise
            
            # C. Model Prediction
            # Input: [x_noisy (3) | mask (1) | masked_context (3)]
            model_input = torch.cat([x_noisy, mask, masked_context], dim=1)
            
            # Predict Noise
            pred_noise = model(model_input, t, c_img, c_text_cond)
            
            loss = F.mse_loss(pred_noise, noise)
            
            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            global_step += 1
            progress_bar.set_postfix(loss=loss.item())
        
        # Save Checkpoint
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            save_path = os.path.join(args.output_dir, f"biomed_ddbm_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pos_dir", type=str, required=True)
    parser.add_argument("--neg_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="biomed_ddbm_output")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    train(args)
