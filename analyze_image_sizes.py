"""Analyze image size distribution to determine best preprocessing strategy"""
from pathlib import Path
import numpy as np
from PIL import Image

def analyze_image_sizes(images_dir):
    """Analyze all image dimensions"""
    images_dir = Path(images_dir)
    sizes = []
    
    print(f"Analyzing {images_dir}...")
    for img_path in sorted(images_dir.glob("*.png")):
        try:
            img = Image.open(img_path)
            sizes.append(img.size)  # (width, height)
        except Exception as e:
            print(f"Error loading {img_path.name}: {e}")
    
    if not sizes:
        print("No images found!")
        return
    
    sizes = np.array(sizes)
    widths, heights = sizes[:, 0], sizes[:, 1]
    
    print(f"\n=== Image Size Distribution ===")
    print(f"Total images: {len(sizes)}")
    print(f"\nWidth statistics:")
    print(f"  Min: {widths.min()}, Max: {widths.max()}")
    print(f"  Mean: {widths.mean():.1f}, Std: {widths.std():.1f}")
    print(f"  Median: {np.median(widths):.1f}")
    
    print(f"\nHeight statistics:")
    print(f"  Min: {heights.min()}, Max: {heights.max()}")
    print(f"  Mean: {heights.mean():.1f}, Std: {heights.std():.1f}")
    print(f"  Median: {np.median(heights):.1f}")
    
    # Aspect ratio analysis
    aspect_ratios = widths / heights
    print(f"\nAspect ratio (W/H) statistics:")
    print(f"  Min: {aspect_ratios.min():.2f}, Max: {aspect_ratios.max():.2f}")
    print(f"  Mean: {aspect_ratios.mean():.2f}, Median: {np.median(aspect_ratios):.2f}")
    
    # Count images smaller/larger than 224
    smaller_224 = np.sum((widths < 224) | (heights < 224))
    larger_224 = np.sum((widths > 224) & (heights > 224))
    mixed_224 = len(sizes) - smaller_224 - larger_224
    
    print(f"\n=== Relative to 224x224 target ===")
    print(f"  Images smaller than 224 in any dimension: {smaller_224}")
    print(f"  Images larger than 224 in both dimensions: {larger_224}")
    print(f"  Mixed size: {mixed_224}")

if __name__ == "__main__":
    images_dir = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/data/brain_tumors/train_images"
    analyze_image_sizes(images_dir)
