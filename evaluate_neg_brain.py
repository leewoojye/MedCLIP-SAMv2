import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, AutoTokenizer

# Load BiomedCLIP model
model_name = "chuhac/BiomedCLIP-vit-bert-hf"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading BiomedCLIP model: {model_name} on {device}")
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Path setup
original_dir = "data/brain_tumors/test_images"
generated_dir = "generated_neg_output/brain_tumors"

# Text Prompts
prompts = ["a brain MRI with a tumor", "a healthy brain MRI"]

# Get list of generated images (assuming filenames match originals)
generated_files = sorted([f for f in os.listdir(generated_dir) if f.endswith(('.png', '.jpg')) and not f.endswith('_mask.png')])

print(f"Found {len(generated_files)} generated images.")
print("-" * 60)
print(f"{'Image ID':<15} | {'Orig Tumor':<12} | {'Orig Healthy':<12} | {'Gen Tumor':<12} | {'Gen Healthy':<12} | {'Success?':<8}")
print("-" * 60)

for filename in generated_files:
    orig_path = os.path.join(original_dir, filename)
    gen_path = os.path.join(generated_dir, filename)
    
    if not os.path.exists(orig_path):
        print(f"Warning: Original image not found for {filename}")
        continue
        
    try:
        image_orig = Image.open(orig_path).convert("RGB")
        image_gen = Image.open(gen_path).convert("RGB")
    except Exception as e:
        print(f"Error loading images for {filename}: {e}")
        continue

    # Prepare inputs
    inputs_orig = processor(text=prompts, images=image_orig, return_tensors="pt", padding=True).to(device)
    inputs_gen = processor(text=prompts, images=image_gen, return_tensors="pt", padding=True).to(device)

    # Inference
    with torch.no_grad():
        outputs_orig = model(**inputs_orig)
        outputs_gen = model(**inputs_gen)
    
    # Calculate probabilities (softmax over prompts)
    probs_orig = outputs_orig.logits_per_image.softmax(dim=1).cpu().numpy()[0]
    probs_gen = outputs_gen.logits_per_image.softmax(dim=1).cpu().numpy()[0]
    
    # Check if "healthy" score is now higher than "tumor" score
    is_success = probs_gen[1] > probs_gen[0]
    success_mark = "YES" if is_success else "NO"

    print(f"{filename:<15} | {probs_orig[0]:.4f}       | {probs_orig[1]:.4f}         | {probs_gen[0]:.4f}      | {probs_gen[1]:.4f}        | {success_mark}")

print("-" * 60)
print("Note: Scores are softmax probabilities between 'tumor' and 'healthy' prompts.")
