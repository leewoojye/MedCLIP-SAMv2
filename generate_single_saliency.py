import sys
import os
import torch
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from PIL import Image

sys.path.append("saliency_maps")
try:
    from scripts.methods import vision_heatmap_iba
except ImportError:
    print("Warning: could not import vision_heatmap_iba. Make sure you run this script from the MedCLIP-SAMv2 root directory.")
    sys.exit(1)

def overlay_heatmap(img, heatmap, colormap=cv2.COLORMAP_JET, alpha=0.5):
    """
    Overlays a heatmap onto an image.
    """
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    img_np = np.array(img.convert('RGB'))
    heatmap_resized = cv2.resize(heatmap_colored, (img_np.shape[1], img_np.shape[0]))
    
    overlay = cv2.addWeighted(img_np, 1 - alpha, heatmap_resized, alpha, 0)
    return Image.fromarray(overlay)

def main():
    parser = argparse.ArgumentParser(description="Generate a saliency map for a single image using BiomedCLIP")
    parser.add_argument("--image", required=True, type=str, help="Path to input image")
    parser.add_argument("--text", required=True, type=str, help="Text prompt to generate saliency for (e.g., 'tumor', 'healthy brain')")
    parser.add_argument("--out", default="saliency_output.png", type=str, help="Output image path")
    parser.add_argument("--layer", type=int, default=9, help="Vision layer for IBA (default 7)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to finetuned model directory (if any)")
    
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading BiomedCLIP model...")
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading finetuned model from: {args.model_path}")
        model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).to(device)
    else:
        print("Loading default BiomedCLIP model from huggingface...")
        model = AutoModel.from_pretrained("chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True).to(device)
        
    # Always load processor and tokenizer from the base HF model since local custom models usually lack processing scripts
    tokenizer_path = "chuhac/BiomedCLIP-vit-bert-hf"
    
    processor = AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    # Monkey-patch config to fix AttributeError in IBA/transformers
    if not hasattr(model.config.text_config, "is_decoder"):
        model.config.text_config.is_decoder = False

    print(f"Loading image: {args.image}")
    try:
        image = Image.open(args.image).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # Preprocess image and text
    image_feat = processor(images=image, return_tensors="pt")['pixel_values'].to(device)
    text_ids = torch.tensor([tokenizer.encode(args.text, add_special_tokens=True)]).to(device)

    # Debug: Check token bounds for CUDA gather error
    print(f"DEBUG text_ids max: {text_ids.max()}, min: {text_ids.min()}, shape: {text_ids.shape}")
    vocab_size = model.config.text_config.vocab_size
    print(f"DEBUG Model vocab size: {vocab_size}")
    
    # Patch: If the finetuned model has a smaller vocab size than the tokenizer, clamp the IDs to prevent CUDA out of bounds
    if text_ids.max() >= vocab_size:
        print(f"Warning: Found tokens ({text_ids.max()}) exceeding model vocab size ({vocab_size}). Clamping them to {vocab_size - 1}.")
        text_ids = torch.clamp(text_ids, max=vocab_size - 1)

    print(f"Generating saliency map for text: '{args.text}'...")
    vmap = vision_heatmap_iba(text_ids, image_feat, model, args.layer, beta=2.0, var=0.3, ensemble=False, progbar=False)

    # Convert to numpy array if it's a tensor
    if isinstance(vmap, torch.Tensor):
        vmap = vmap.detach().cpu().numpy()
        
    vmap = np.array(vmap)
    
    # Normalize heatmap between 0 and 1
    if vmap.max() > vmap.min():
        vmap = (vmap - vmap.min()) / (vmap.max() - vmap.min())

    # Generate output filenames
    base, ext = os.path.splitext(args.out)
    overlay_path = args.out
    raw_path = f"{base}_raw{ext}"

    # Save raw heatmap
    raw_vmap_resized = cv2.resize(vmap, (image.size[0], image.size[1]), interpolation=cv2.INTER_NEAREST)
    plt.imsave(raw_path, raw_vmap_resized, cmap='jet')

    # Save overlay image
    result = overlay_heatmap(image, vmap)
    result.save(overlay_path)
    
    print(f"Done!")
    print(f"Saved overlay image to: {overlay_path}")
    print(f"Saved raw heatmap to:   {raw_path}")

if __name__ == "__main__":
    main()
