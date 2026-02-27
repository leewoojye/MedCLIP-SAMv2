import os
import argparse
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load the Finetuned UNet natively
    print(f"Loading finetuned merged U-Net from: {args.lora_path}")
    unet = UNet2DConditionModel.from_pretrained(args.lora_path, torch_dtype=torch.float32, low_cpu_mem_usage=False)

    # 2. Build Pipeline
    print(f"Loading Base Pipeline: {args.base_model}")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.base_model,
        unet=unet,
        torch_dtype=torch.float32, 
        safety_checker=None,
        low_cpu_mem_usage=False
    ).to(device)

    # 3. Load input images
    init_image = Image.open(args.image).convert("RGB").resize((512, 512))
    mask_image = Image.open(args.mask).convert("RGB").resize((512, 512))

    # 4. Generate Output
    print(f"Generating image with prompt: '{args.prompt}'")
    
    # Standard diffuser prompt injection
    image = pipe(
        prompt=args.prompt,
        image=init_image,
        mask_image=mask_image,
        num_inference_steps=50,
        guidance_scale=7.5 # Standard CFG scale
    ).images[0]

    import numpy as np
    import cv2
    image.save(args.out)
    print(f"Saved successful generation to: {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=str, help="Path to input image")
    parser.add_argument("--mask", required=True, type=str, help="Path to inpainting mask")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--lora_path", type=str, default="zero_shot_translation/lora_weights_natural")
    parser.add_argument("--prompt", type=str, default="a healthy brain")
    parser.add_argument("--out", type=str, default="zero_shot_translation/output_lora_brain.png")
    args = parser.parse_args()
    main(args)
