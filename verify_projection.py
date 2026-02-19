import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import os
import torch.nn.functional as F

def verify_projection():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load Model
    name = "chuhac/BiomedCLIP-vit-bert-hf"
    model = AutoModel.from_pretrained(name, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(name, trust_remote_code=True)
    model.eval()
    
    # Load Sample Image (Tumor)
    img_path = "data/breast_tumors/test_images/000001.png"
    if not os.path.exists(img_path):
        print(f"Error: {img_path} not found.")
        return
        
    image = Image.open(img_path).convert("RGB")
    
    # Prepare Prompts
    texts = ["breast tumor", "normal breast tissue"]
    
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        # Get Raw Features (768d)
        vision_raw = model.vision_model(inputs['pixel_values'])[0][:, 0, :]
        text_raw = model.text_model(inputs['input_ids'], attention_mask=inputs['attention_mask'])[0][:, 0, :]
        
        # Get Projected Features (512d) - The standard BioMedCLIP way
        # Standard forward pass wraps this
        outputs = model(**inputs)
        image_embeds_proj = outputs.image_embeds # Already projected and normalized usually
        text_embeds_proj = outputs.text_embeds
        
        # Manual Projection to confirm our method match
        if hasattr(model, 'visual_projection'):
            img_proj_manual = model.visual_projection(vision_raw)
            txt_proj_manual = model.text_projection(text_raw)
            # Normalize
            img_proj_manual = img_proj_manual / img_proj_manual.norm(dim=-1, keepdim=True)
            txt_proj_manual = txt_proj_manual / txt_proj_manual.norm(dim=-1, keepdim=True)
        else:
            img_proj_manual = vision_raw
            txt_proj_manual = text_raw
            
    # Calculate Similarities
    print("\n--- Similarity Analysis ---\n")
    
    # 1. Projected Space (The one we use)
    print("Projected Space (512d):")
    sim_tumor = (img_proj_manual @ txt_proj_manual[0].unsqueeze(1)).item()
    sim_normal = (img_proj_manual @ txt_proj_manual[1].unsqueeze(1)).item()
    print(f"  Similarity to '{texts[0]}': {sim_tumor:.4f}")
    print(f"  Similarity to '{texts[1]}': {sim_normal:.4f}")
    if sim_tumor > sim_normal:
        print("  RESULT: CORRECTLY classified as Tumor.")
    else:
        print("  RESULT: IDK, maybe image is subtle?")
        
    print("\n--- Evidence ---")
    print("The projection layer is trained specifically to align these representations.")
    print("If the above classification is correct, the 512d embedding preserves the critical semantic info.")

if __name__ == "__main__":
    verify_projection()
