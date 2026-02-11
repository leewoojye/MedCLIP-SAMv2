
import sys
import torch
import os

# Add OpenCLIP to path
sys.path.append(os.path.join(os.getcwd(), 'biomedclip_finetuning/open_clip/src'))
import open_clip

def print_module_structure():
    model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    print(f"Creating model: {model_name}")
    try:
        model = open_clip.create_model(model_name, pretrained=False)
        print("Model created.")
        
        print("\n--- Visual Module Analysis ---")
        visual = model.visual
        print(f"Visual type: {type(visual)}")
        if hasattr(visual, 'trunk'):
            print("Visual has 'trunk'")
        if hasattr(visual, 'head'):
            print("Visual has 'head'")
            print(f"Visual head: {visual.head}")
        if hasattr(visual, 'proj'):
            print(f"Visual proj: {visual.proj}")
            
        print("\n--- Text Module Analysis ---")
        text = model.text
        print(f"Text type: {type(text)}")
        if hasattr(text, 'proj'):
            print(f"Text proj: {text.proj}")
        if hasattr(text, 'text_projection'):
            print(f"Text text_projection: {text.text_projection}")
        if hasattr(model, 'text_projection'):
            print(f"Model text_projection: {model.text_projection}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print_module_structure()
