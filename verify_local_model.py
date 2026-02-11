
import torch
# Add current dir to path to find biomed_model
import sys
import os
sys.path.append(os.getcwd())

try:
    from biomed_model.modeling_biomed_clip import BiomedCLIPModel
    from biomed_model.configuration_biomed_clip import BiomedCLIPConfig

    print("Successfully imported BiomedCLIPModel from local file.")
    
    # Try loading the model weights into this class
    # We use from_pretrained but point to the hugginface repo, 
    # BUT we pass `config` that is initialized from our local class?
    # Or just use `BiomedCLIPModel.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf")`
    # This might fail if the repo's config.json says "auto_map" points to remote code.
    
    # Better: Load config, remove auto_map/remote code ref? 
    # Actually, from_pretrained on the class should work if weights match.
    
    print("Loading model weights...")
    model = BiomedCLIPModel.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf")
    print("Model loaded successfully!")
    print(f"Vision model: {type(model.vision_model)}")
    print(f"Text model: {type(model.text_model)}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
