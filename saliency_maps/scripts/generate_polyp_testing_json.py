"""
Generate polyp_testing.json mapping image filenames to descriptive polyp prompts.
"""
import os
import json
import random
from pathlib import Path

PROMPTS = [
    "A colonoscopy image showing a sessile colorectal polyp with erythematous mucosal protrusion.",
    "A gastrointestinal endoscopy view of a pedunculated polyp with smooth lobulated contours and a distinct stalk.",
    "An endoscopic image revealing an adenomatous polyp with irregular surface and disrupted mucosal pattern.",
    "A colonoscopy frame showing a serrated colorectal polyp with pale mucosa and indistinct borders.",
    "An endoscopy image displaying a flat elevated lesion consistent with a non-pedunculated colorectal polyp.",
    "A GI endoscopy image showing a hyperplastic polyp with smooth hemispheric contour and uniform coloration.",
    "A colonoscopy image of a villous adenoma with frond-like surface architecture and contact bleeding.",
    "An endoscopic view showing a diminutive colorectal polyp with well-circumscribed mucosal elevation.",
    "A colonoscopy image showing a depressed lesion suspicious for an early neoplastic colorectal polyp.",
    "An endoscopy image displaying irregular vascular and pit patterns consistent with an adenomatous colorectal polyp."
]

def main():
    base = Path('/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2')
    images_dir = base / 'data/polyp/test_images'
    output_path = base / 'saliency_maps/text_prompts/polyp_testing.json'

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    if not image_files:
        raise RuntimeError("No test images found for polyp dataset.")

    random.seed(42)
    mapping = {}
    for fname in image_files:
        prompt = random.choice(PROMPTS)
        mapping[fname] = prompt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"Wrote {len(mapping)} entries to {output_path}")

if __name__ == '__main__':
    main()
