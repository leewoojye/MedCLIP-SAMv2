import os
import torch
import sys
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoTokenizer
import numpy as np

def main():
    # Load BiomedCLIP model
    base_model_name = "chuhac/BiomedCLIP-vit-bert-hf"
    
    # You can also use the finetuned model checkpoint if preferred, 
    # but the base model is usually very good for general body parts
    # model_checkpoint = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/saliency_maps/model"
    model_checkpoint = base_model_name
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from: {model_checkpoint} on {device}")
    
    model = AutoModel.from_pretrained(model_checkpoint, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    
    # Image Path
    image_path = sys.argv[1] if len(sys.argv) > 1 else "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/321 copy.png"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
        
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
        
    # Define candidate anatomical locations / modalities
    domains = [
        "a brain MRI",
        "a breast ultrasound",
        "a colonoscopy image",
        "a chest X-ray",
        "a skin lesion dermoscopy",
        "a retinal fundus image",
        "a pelvic MRI",
        "an abdominal CT scan"
    ]
    
    print("-" * 60)
    print(f"Evaluating image: {os.path.basename(image_path)}")
    print("-" * 60)
    print("Candidate Domains:")
    
    # Process inputs
    inputs = processor(text=domains, images=image, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Get probabilities
    probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()[0]
    
    # Sort results
    results = list(zip(domains, probs))
    results.sort(key=lambda x: x[1], reverse=True)
    
    # Print results
    for domain, prob in results:
        print(f"{domain:<30} | Confidence: {prob*100:.2f}%")
        
    print("-" * 60)
    print(f"\n=> 🎯 Best Guess: {results[0][0]} (Confidence: {results[0][1]*100:.2f}%)")

if __name__ == "__main__":
    main()
