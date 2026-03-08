import cv2
import glob
import os
import shutil
import numpy as np
import subprocess

# Directories
input_base_dir = '/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/Dataset_BUSI_with_GT'
output_base_dir = '/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/BUSI_PREPROCESSED'
mask_base_dir = '/tmp/busi_robust_masks'
classes = ['benign', 'malignant', 'normal']

def get_robust_mask(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge detection to find all sharp markings (white or non-white)
    edges = cv2.Canny(img_gray, 100, 200)
    
    # 2. Strict thresholding for very bright markers (almost white)
    _, bright_mask = cv2.threshold(img_gray, 245, 255, cv2.THRESH_BINARY)
    
    # 3. Combine edges and bright pixels using closing
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
    combined = cv2.bitwise_or(closed_edges, bright_mask)
    
    # 4. Filter by shape (Connected Components)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(combined, connectivity=8)
    
    final_mask = np.zeros_like(img_gray)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        aspect_ratio = float(w) / h if h > 0 else 0
        density = float(area) / (w * h) if (w * h) > 0 else 0
        
        # Categorize as annotation if it matches specific geometric profiles
        is_annotation = False
        if area < 500: # Small text/pips
            is_annotation = True
        elif (aspect_ratio > 8 or aspect_ratio < 0.12) and density < 0.4: # Thin lines (bbox/dotted)
            is_annotation = True
        elif area < 1200 and density < 0.25: # Box outlines
            is_annotation = True
            
        if is_annotation:
            final_mask[labels == i] = 255
            
    # 5. Dilation (Ensure full coverage of digital overlay)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    final_mask = cv2.dilate(final_mask, kernel_dilate, iterations=1)
    
    return final_mask

try:
    for cls in classes:
        in_dir = os.path.join(input_base_dir, cls)
        out_dir = os.path.join(output_base_dir, cls)
        mask_dir = os.path.join(mask_base_dir, cls)
        
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        images = glob.glob(os.path.join(in_dir, '*.png'))
        image_files = [img for img in images if 'mask' not in os.path.basename(img)]
        mask_files = [img for img in images if 'mask' in os.path.basename(img)]

        # 1. Copy GT Masks directly
        print(f"[{cls}] Copying GT masks...")
        for m in mask_files:
            shutil.copy2(m, os.path.join(out_dir, os.path.basename(m)))

        # 2. Generate robust inpainting masks
        print(f"[{cls}] Generating AI inpainting masks...")
        for img_path in image_files:
            basename = os.path.basename(img_path)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            
            mask = get_robust_mask(img_bgr)
            cv2.imwrite(os.path.join(mask_dir, basename), mask)
            
        # 3. Batch run LaMa AI via iopaint
        print(f"[{cls}] Running LaMa batch inpainting...")
        cmd = [
            "python", "-m", "iopaint", "run",
            "--model", "lama",
            "--image", in_dir,
            "--mask", mask_dir,
            "--output", out_dir,
            "--device", "cuda"
        ]
        subprocess.run(cmd, check=True)

    print("Success: BUSI dataset preprocessed with AI Inpainting.")

except Exception as e:
    print(f"Critical error during batch processing: {e}")
