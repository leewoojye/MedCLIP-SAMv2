
import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, get_scheduler, AutoConfig

# Local imports
import sys
sys.path.append(os.getcwd())
try:
    from biomed_model.modeling_biomed_clip import BiomedCLIPModel
except ImportError:
    print("Error importing local model. Ensure 'biomed_model' directory exists.")
    raise

from biomed_dataset import EgoBridgeDataset
from biomed_loss import EgoBridgeLoss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune BiomedCLIP with EgoBridge strategy")
    
    # Dataset and Model
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset root (e.g. Dataset_BUSI_with_GT)")
    parser.add_argument("--output-dir", type=str, default="./output", help="Directory to save checkpoints")
    parser.add_argument("--model-name", type=str, default="chuhac/BiomedCLIP-vit-bert-hf")
    parser.add_argument("--pretrained-checkpoint", type=str, default=None, help="Path to resume from checkpoint")
    
    # Training
    parser.add_argument("--stage", type=str, choices=["stage1", "stage2"], required=True, help="EgoBridge stage")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--wd", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    
    # Loss Params
    parser.add_argument("--sinkhorn-eps", type=float, default=0.05)
    parser.add_argument("--contrastive-lambda", type=float, default=1.0)
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset
    print(f"Initializing Dataset for {args.stage}...")
    dataset = EgoBridgeDataset(root_dir=args.data_dir, stage=args.stage)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # 2. Model
    print("Loading Model...")
    # Load default config first?
    if args.pretrained_checkpoint:
        print(f"Loading weights from {args.pretrained_checkpoint}")
        # Assuming checkpoint is a full HF model directory or just state_dict?
        # If directory, use from_pretrained. If .pt file, load_state_dict.
        if os.path.isdir(args.pretrained_checkpoint):
             model = BiomedCLIPModel.from_pretrained(args.pretrained_checkpoint)
        elif args.pretrained_checkpoint.endswith('.pt') or args.pretrained_checkpoint.endswith('.bin'):
             # Load base config then weights
             model = BiomedCLIPModel.from_pretrained(args.model_name)
             state_dict = torch.load(args.pretrained_checkpoint, map_location='cpu')
             model.load_state_dict(state_dict, strict=False) # reckless loading?
        else:
             model = BiomedCLIPModel.from_pretrained(args.model_name)
    else:
        model = BiomedCLIPModel.from_pretrained(args.model_name)
    
    model.to(device)
    model.train()
    
    # Freeze Text Encoder? EgoBridge usually freezes text encoder or co-trains?
    # BiomedCLIP paper freezes text usually. Let's freeze text encoder to save memory and stability.
    for param in model.text_model.parameters():
        param.requires_grad = False
    
    # 3. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    # Scheduler
    num_training_steps = args.epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    
    # 4. Loss
    criterion = EgoBridgeLoss(sinkhorn_eps=args.sinkhorn_eps, contrastive_lambda=args.contrastive_lambda).to(device)
    
    # 5. Training Loop
    print(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            # Forward Logic based on Stage
            # Dataset returns:
            # Stage 1: image1, mask1, image2, mask2
            # Stage 2: image_pos, mask_pos, image_neg
            
            if args.stage == 'stage1':
                # Move inputs to device
                img1 = batch['image1'].to(device)
                mask1 = batch['mask1'].to(device)
                img2 = batch['image2'].to(device)
                mask2 = batch['mask2'].to(device)
                
                # Forward Pass (Vision Only)
                # Need features!
                # Using get_image_features from HF model?
                # Wait, get_image_features returns pooled output by default?
                # We need features sequence for Sinkhorn!
                # Let's inspect model.vision_model forward.
                # It returns BaseModelOutputWithPooling.
                # output_hidden_states=False default.
                
                # We need last_hidden_state (B, Seq, Dim).
                
                # Call vision model directly!
                out1 = model.vision_model(pixel_values=img1, return_dict=True)
                out2 = model.vision_model(pixel_values=img2, return_dict=True)
                
                feat1 = out1.last_hidden_state # [B, 197, 768]
                feat2 = out2.last_hidden_state
                
                # Apply projection? 
                # BiomedCLIP has visual_projection (Linear).
                # Usually applied to POOLED output.
                # Should we apply it to sequence?
                # Typically dense alignment happens in shared space.
                # But projection is (768 -> 512).
                # If we use raw 768, it's fine for internal consistence, but text is 512.
                # Sinkhorn ensures Pos-Pos alignment.
                # Let's align in Projected Space if possible?
                # But projection is linear.
                # Let's project token-wise.
                
                feat1_proj = model.visual_projection(feat1) # [B, 197, 512]
                feat2_proj = model.visual_projection(feat2)
                
                loss = criterion.forward_stage1(feat1_proj, mask1, feat2_proj, mask2)
                
            elif args.stage == 'stage2':
                img_pos = batch['image_pos'].to(device)
                mask_pos = batch['mask_pos'].to(device)
                img_neg = batch['image_neg'].to(device)
                
                out_pos = model.vision_model(pixel_values=img_pos, return_dict=True)
                out_neg = model.vision_model(pixel_values=img_neg, return_dict=True)
                
                feat_pos = out_pos.last_hidden_state
                feat_neg = out_neg.last_hidden_state
                
                feat_pos_proj = model.visual_projection(feat_pos)
                feat_neg_proj = model.visual_projection(feat_neg)
                
                loss = criterion.forward_stage2(feat_pos_proj, mask_pos, feat_neg_proj)
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            total_loss += loss.item()
            
            if step % 10 == 0:
                print(f"Epoch {epoch+1}/{args.epochs} - Step {step}/{len(dataloader)} - Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} Completed. Avg Loss: {avg_loss:.4f}")
        
        # Save Checkpoint
        save_path = os.path.join(args.output_dir, f"epoch_{epoch+1}")
        os.makedirs(save_path, exist_ok=True)
        # Save full model (including text, though frozen) to serve as valid checkpoint
        model.save_pretrained(save_path)
        
        # Copy necessary model files for AutoModel loading
        import shutil
        try:
            shutil.copy("biomed_model/modeling_biomed_clip.py", save_path)
            shutil.copy("biomed_model/configuration_biomed_clip.py", save_path)
            # processing_biomed_clip might be optional depending on usage but good to have
            if os.path.exists("biomed_model/processing_biomed_clip.py"):
                shutil.copy("biomed_model/processing_biomed_clip.py", save_path)
        except Exception as e:
            print(f"Warning: Could not copy model files to checkpoint: {e}")
            
        print(f"Saved checkpoint to {save_path}")

    print("Training Completed.")

if __name__ == "__main__":
    main()
