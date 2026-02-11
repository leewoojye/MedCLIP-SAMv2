import sys
sys.path.append("biomedclip_finetuning/open_clip/src")
import open_clip
import torch

def check_default_load():
    model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    print(f"Creating model {model_name} WITHOUT explicit pretrained argument...")
    
    try:
        # Create model without overriding pretrained
        model, _, _ = open_clip.create_model_and_transforms(model_name)
        
        # Check if weights are random or trained
        # We check a specific weight we know shouldn't be zero or essentially random small norms if trained?
        # A good check is valid norms.
        # But easier: Check if logit_scale is the default (usually log(1/0.07) ~ 2.65) or trained value.
        # BioMedCLIP logit_scale might be different.
        
        logit_scale = model.logit_scale.item()
        print(f"Logit scale: {logit_scale}")
        
        # Check a visual weight
        first_weight = model.visual.trunk.patch_embed.proj.weight.flatten()[:5]
        print(f"Visual sample: {first_weight}")
        
        # If it crashed, we won't get here.
        # If it succeeds, it means it found necessary files in cache or hub.
        print("Success! Model loaded with default mechanism.")
        
    except Exception as e:
        print(f"Failed to load default: {e}")

if __name__ == "__main__":
    check_default_load()
