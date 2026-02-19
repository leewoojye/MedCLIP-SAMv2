import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoTokenizer
import numpy as np

# Load BiomedCLIP model
model_name = "chuhac/BiomedCLIP-vit-bert-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading BiomedCLIP model: {model_name} on {device}")
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Image Path
image_path = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/synpic31971.jpg"

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
    "Brain": ["a brain MRI with a tumor", "a healthy brain MRI"],
    "Breast": ["a breast ultrasound with a tumor", "a healthy breast ultrasound"],
    "Polyp": ["a colonoscopy image with a polyp", "a healthy colonoscopy image"]
}

print("-" * 60)
print(f"Evaluating image: {os.path.basename(image_path)}")
print("-" * 60)
print(f"{'Domain':<10} | {'Tumor (Pos)':<12} | {'Healthy (Neg)':<12} | {'Prediction':<10}")
print("-" * 60)

for domain, prompts in prompt_sets.items():
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
    tumor_score = probs[0]
    healthy_score = probs[1]
    
    prediction = "Tumor" if tumor_score > healthy_score else "Healthy"
    
    print(f"{domain:<10} | {tumor_score:.4f}       | {healthy_score:.4f}         | {prediction}")

print("-" * 60)
print("Note: Positive prompt is index 0 (Tumor/Polyp), Negative prompt is index 1 (Healthy).")
