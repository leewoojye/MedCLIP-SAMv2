import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoTokenizer
import numpy as np

# Load BiomedCLIP model
base_model_name = "chuhac/BiomedCLIP-vit-bert-hf"
model_checkpoint = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/saliency_maps/model"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading MedCLIP-SAM model from: {model_checkpoint} on {device}")
model = AutoModel.from_pretrained(model_checkpoint, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

import sys

# Image Path
image_path = sys.argv[1] if len(sys.argv) > 1 else "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/zero_shot_translation/output_text_alpha_4.0.png"

if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
    exit(1)

try:
    image = Image.open(image_path).convert("RGB")
except Exception as e:
    print(f"Error loading image: {e}")
    exit(1)

# Prompt Sets
prompt_sets = {
    "Brain": ["tumor", "healthy, normal"],
    "Breast": ["a breast ultrasound with a tumor", "a healthy breast ultrasound"],
    "Polyp": ["a colonoscopy image with a polyp", "a healthy colonoscopy image"]
}

print("-" * 60)
print(f"Evaluating image: {os.path.basename(image_path)}")
print("-" * 60)
print(f"{'Domain':<10} | {'Tumor (Pos)':<28} | {'Healthy (Neg)':<28} | {'Prediction':<10}")
print("-" * 85)

for domain, prompts in prompt_sets.items():
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits_per_image.cpu().numpy()[0]
    probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
    
    tumor_logit = logits[0]
    healthy_logit = logits[1]
    
    tumor_score = probs[0]
    healthy_score = probs[1]
    
    prediction = "Tumor" if tumor_score > healthy_score else "Healthy"
    
    print(f"{domain:<10} | Prob: {tumor_score:.4f} (Logit: {tumor_logit:>7.4f}) | Prob: {healthy_score:.4f} (Logit: {healthy_logit:>7.4f}) | {prediction}")


print("-" * 60)
print("Note: Positive prompt is index 0 (Tumor/Polyp), Negative prompt is index 1 (Healthy).")
