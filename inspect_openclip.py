
import sys
import torch
import os

# Add OpenCLIP to path
sys.path.append(os.path.join(os.getcwd(), 'biomedclip_finetuning/open_clip/src'))
import open_clip

def inspect_checkpoint(ckpt_path):
    print(f"Inspecting checkpoint: {ckpt_path}")
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print("Checkpoint keys:", ckpt.keys())
        if 'state_dict' in ckpt:
            print("State dict keys sample:", list(ckpt['state_dict'].keys())[:10])
        else:
            print("State dict keys sample:", list(ckpt.keys())[:10])
    except Exception as e:
        print(f"Error loading checkpoint: {e}")

def inspect_model():
    print("Inspecting OpenCLIP model structure...")
    model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
    try:
        model = open_clip.create_model(model_name, pretrained=False)
        print("Model class:", type(model))
        print("Model attributes:", dir(model))
        if hasattr(model, 'visual'):
            print("Visual module attributes:", dir(model.visual))
            if hasattr(model.visual, 'transformer'):
                print("Visual transformer attributes:", dir(model.visual.transformer))
        else:
            print("No 'visual' attribute found.")
    except Exception as e:
        print(f"Error creating model: {e}")

if __name__ == "__main__":
    ckpt_path = "./logs_finetuning/egobridge_stage2_pos_neg_20260211_115516/checkpoints/epoch_10.pt"
    inspect_checkpoint(ckpt_path)
    inspect_model()
