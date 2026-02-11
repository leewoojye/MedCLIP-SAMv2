
from transformers import AutoConfig, AutoModel
import torch
import torch.nn as nn

try:
    print("Loading config...")
    config = AutoConfig.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
    print(f"Config architectural: {config.architectures}")
    print(f"Config model_type: {config.model_type}")
    
    # Try loading model again with more imports available
    print("Loading model...")
    model = AutoModel.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
    print("Model loaded successfully.")
    print(f"Has vision_model: {hasattr(model, 'vision_model')}")
    print(f"Has text_model: {hasattr(model, 'text_model')}")

except Exception as e:
    print(f"Error: {e}")
    # Print what the error really is
    import traceback
    traceback.print_exc()
