import torch
from transformers import AutoModel, AutoProcessor

name = "chuhac/BiomedCLIP-vit-bert-hf"
model = AutoModel.from_pretrained(name, trust_remote_code=True)

print("Model Class:", type(model).__name__)
print(model)

# Check dummy forward
dummy_img = torch.randn(1, 3, 224, 224)
dummy_text = ["test"]
processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)

vision_out = model.vision_model(dummy_img)
text_out = model.text_model(**processor(text=dummy_text, return_tensors="pt"))

print("Vision Last Hidden State:", vision_out[0].shape)
print("Text Last Hidden State:", text_out[0].shape)

if hasattr(model, 'visual_projection'):
    print("Has visual_projection")
    print(model.visual_projection)
elif hasattr(model, 'vision_proj'):
    print("Has vision_proj")
    
if hasattr(model, 'text_projection'):
    print("Has text_projection")
    print(model.text_projection)
elif hasattr(model, 'text_proj'):
    print("Has text_proj")
