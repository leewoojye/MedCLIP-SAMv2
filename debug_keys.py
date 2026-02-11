import torch
import sys
import os

# Add the source directory to path to import open_clip
sys.path.append("biomedclip_finetuning/open_clip/src")
import open_clip

def check_keys():
    checkpoint_path = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/saliency_maps/model/pytorch_model.bin"
    model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        ckpt_keys = list(state_dict.keys())
        print(f"Checkpoint has {len(ckpt_keys)} keys.")
        print("Sample Checkpoint Keys:")
        for k in ckpt_keys[:5]: print(f"  {k}")
        for k in ckpt_keys:
            if "vision_model.encoder.layer.0" in k:
                print(f"  {k}")
                break
        for k in ckpt_keys:
            if "text_model" in k:
                print(f"  {k}")
                break

    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    print(f"\nCreating model {model_name}...")
    try:
        model, _, _ = open_clip.create_model_and_transforms(model_name)
        model_keys = list(model.state_dict().keys())
        print(f"Model has {len(model_keys)} keys.")
        print("Sample Model Keys:")
        for k in model_keys[:5]: print(f"  {k}")
        
        print("Visual Keys:")
        for k in model_keys:
            if "visual.trunk.blocks.0" in k:
                print(f"  {k}")
                break
                
        print("Text Keys:")
        text_keys = [k for k in model_keys if "text" in k]
        for k in text_keys[:3]: print(f"  {k}")
        
    except Exception as e:
        print(f"Error creating model: {e}")

if __name__ == "__main__":
    check_keys()
