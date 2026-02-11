
import os
import sys
import torch
import open_clip

# Add open_clip src to path
sys.path.append(os.path.abspath("biomedclip_finetuning/open_clip/src"))

from open_clip import create_model_and_transforms, create_model

def debug_model():
    model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    print(f"Creating model: {model_name}")
    
    try:
        model, _, _ = create_model_and_transforms(model_name, output_dict=True)
        print("Model created successfully.")
    except Exception as e:
        print(f"Error creating model: {e}")
        return

    print(f"\nModel class: {type(model)}")
    print(f"Model visual class: {type(model.visual)}")
    
    # Check output_tokens attribute
    has_output_tokens = hasattr(model.visual, 'output_tokens')
    print(f"hasattr(model.visual, 'output_tokens'): {has_output_tokens}")
    
    if has_output_tokens:
        print(f"Initial model.visual.output_tokens: {getattr(model.visual, 'output_tokens', 'ERROR')}")
        
        # Test modification
        model.visual.output_tokens = True
        print(f"Set model.visual.output_tokens = True. New value: {model.visual.output_tokens}")
        
        # Test forward pass with dummy input
        print("\nTesting forward pass...")
        dummy_image = torch.randn(1, 3, 224, 224)
        
        try:
            output = model.visual(dummy_image)
            print(f"Output type: {type(output)}")
            if isinstance(output, tuple):
                print(f"Output tuple length: {len(output)}")
                print(f"Output[0] shape (pooled): {output[0].shape}")
                print(f"Output[1] shape (tokens): {output[1].shape}")
            else:
                print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Error during forward pass: {e}")
            import traceback
            traceback.print_exc()

    else:
        print("model.visual DOES NOT have 'output_tokens' attribute.")
        print(f"Dir(model.visual): {dir(model.visual)}")

if __name__ == "__main__":
    debug_model()
