import torch
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from PIL import Image
device = 'cuda'
print("Loading Model...")
model = AutoModel.from_pretrained("/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/saliency_maps/model", trust_remote_code=True).to(device)
if not hasattr(model.config.text_config, "is_decoder"):
    model.config.text_config.is_decoder = False
processor = AutoProcessor.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True)
image = Image.open('32.png').convert('RGB')
image_feat = processor(images=image, return_tensors='pt')['pixel_values'].to(device)
text_ids = torch.tensor([tokenizer.encode("tumor", add_special_tokens=True)]).to(device)
print("Forwarding text...")
txt_out = model.get_text_features(text_ids)
print("Text Success! Forwarding Vision...")
vis_out = model.get_image_features(image_feat)
print("Vision Success!")
