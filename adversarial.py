import sys
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel
import numpy as np

model_name = "chuhac/BiomedCLIP-vit-bert-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading BiomedCLIP model: {model_name} on {device}")
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

for param in model.parameters():
    param.requires_grad = False

image_path = "data/brain_tumors/val_images/1013.png"
original_image = Image.open(image_path).convert("RGB")
prompts = ["a brain MRI with a tumor", "a healthy brain MRI"]

# Process inputs once
inputs = processor(text=prompts, images=original_image, return_tensors="pt", padding=True)
pixel_values = inputs.pixel_values.to(device)
pixel_values.requires_grad = True

input_ids = inputs.input_ids.to(device)
attention_mask = inputs.attention_mask.to(device)

optimizer = torch.optim.Adam([pixel_values], lr=0.01)

print("Starting adversarial optimization...")
for i in range(101):
    optimizer.zero_grad()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)[0]
    
    tumor_score = probs[0]
    healthy_score = probs[1]
    
    # We want to maximize healthy_score, meaning minimize -healthy_score or minimize tumor_score
    loss = tumor_score
    
    if healthy_score > 0.6:
        print(f"Success at step {i}: Healthy {healthy_score.item():.4f}, Tumor {tumor_score.item():.4f}")
        break
        
    loss.backward()
    optimizer.step()
    
    if i % 10 == 0:
        print(f"Step {i}: Healthy {healthy_score.item():.4f}, Tumor {tumor_score.item():.4f}")

# Convert back to image
image_mean = processor.image_processor.image_mean
image_std = processor.image_processor.image_std

adv_image = pixel_values[0].detach().cpu().numpy()
adv_image = adv_image.transpose(1, 2, 0)
adv_image = adv_image * np.array(image_std) + np.array(image_mean)
adv_image = np.clip(adv_image, 0, 1) * 255
adv_image = adv_image.astype(np.uint8)

Image.fromarray(adv_image).save("adversarial_healthy.png")
print("Saved adversarial image to adversarial_healthy.png")
