import cv2
import glob
import os
import shutil
import numpy as np
import subprocess

input_base_dir = '/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/Dataset_BUSI_with_GT'
output_base_dir = '/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/BUSI_PREPROCESSED_LAMA'
mask_base_dir = '/tmp/busi_lama_masks'
classes = ['benign', 'malignant', 'normal']

def get_annotation_mask(img_gray):
    _, thresh = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    mask = np.zeros_like(thresh)
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        extent = area / (w * h) if (w * h) > 0 else 0
        if area < 800 and extent < 0.6: 
            mask[labels == i] = 255
        elif w > 15 and h > 15 and extent < 0.3:
            mask[labels == i] = 255
            
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    return mask_dilated

try:
    for cls in classes:
        in_dir = os.path.join(input_base_dir, cls)
        out_dir = os.path.join(output_base_dir, cls)
        mask_dir = os.path.join(mask_base_dir, cls)
        
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        images = glob.glob(os.path.join(in_dir, '*.png'))
        
        # 1. Generate all masks
        print(f"Generating masks for {cls}...")
        for img_path in images:
            basename = os.path.basename(img_path)
            
            if 'mask' in basename:
                shutil.copy2(img_path, os.path.join(out_dir, basename))
                continue
                
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
                
            img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            mask = get_annotation_mask(img_gray)
            
            mask_path = os.path.join(mask_dir, basename)
            cv2.imwrite(mask_path, mask)
            
        # 2. Run iopaint on this class folder
        print(f"Running iopaint for {cls}...")
        cmd = [
            "python", "-m", "iopaint", "run",
            "--model", "lama",
            "--image", in_dir,
            "--mask", mask_dir,
            "--output", out_dir,
            "--device", "cuda"
        ]
        subprocess.run(cmd, check=True)

except Exception as e:
    print(f"Error occurred: {e}")

print("LaMa preprocessing complete. Results saved in", output_base_dir)
