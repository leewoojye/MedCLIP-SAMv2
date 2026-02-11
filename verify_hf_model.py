
from transformers import AutoModel
import torch

try:
    print("Loading model...")
    model = AutoModel.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
    print(f"Model type: {type(model)}")
    print(f"Has vision_model: {hasattr(model, 'vision_model')}")
    print(f"Has text_model: {hasattr(model, 'text_model')}")
    if hasattr(model, 'vision_model'):
        print(f"Vision model type: {type(model.vision_model)}")
    if hasattr(model, 'text_model'):
        print(f"Text model type: {type(model.text_model)}")
        
except Exception as e:
    print(f"Error: {e}")
