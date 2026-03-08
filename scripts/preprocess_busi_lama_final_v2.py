import cv2
import glob
import os
import shutil
import numpy as np
import subprocess

# Directories
input_base_dir = '/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/Dataset_BUSI_with_GT'
output_base_dir = '/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/BUSI_PREPROCESSED'
tmp_root = '/tmp/busi_iopaint_staging'
classes = ['benign', 'malignant', 'normal']

def get_robust_mask_v2(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Mask for very bright pixels (digital text/dots) - Processed separately to avoid merging with tissue
    _, bright_thresh = cv2.threshold(img_gray, 245, 255, cv2.THRESH_BINARY)
    num_labels_b, labels_b, stats_b, _ = cv2.connectedComponentsWithStats(bright_thresh, connectivity=8)
    mask_bright = np.zeros_like(img_gray)
    for i in range(1, num_labels_b):
        if stats_b[i, cv2.CC_STAT_AREA] < 1000: 
            mask_bright[labels_b == i] = 255
            
    # 2. Mask for geometric edges (boxes/dotted lines)
    edges = cv2.Canny(img_gray, 50, 150)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_close)
    num_labels_e, labels_e, stats_e, _ = cv2.connectedComponentsWithStats(closed_edges, connectivity=8)
    mask_edges = np.zeros_like(img_gray)
    for i in range(1, num_labels_e):
        x, y, w, h, area = stats_e[i]
        aspect_ratio = float(w) / h if h > 0 else 0
        density = float(area) / (w * h) if (w * h) > 0 else 0
        if (aspect_ratio > 8 or aspect_ratio < 0.12) and density < 0.4:
            mask_edges[labels_e == i] = 255
        elif area < 1200 and density < 0.25:
            mask_edges[labels_e == i] = 255

    # 3. Combine and Dilate
    combined = cv2.bitwise_or(mask_bright, mask_edges)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    final_mask = cv2.dilate(combined, kernel_dilate, iterations=1)
    return final_mask

def sanitize_filename(filename):
    return filename.replace(" ", "_").replace("(", "").replace(")", "")

try:
    for cls in classes:
        in_dir = os.path.join(input_base_dir, cls)
        out_dir = os.path.join(output_base_dir, cls)
        
        # Staging directories for iopaint (sanitized names)
        stage_img_dir = os.path.join(tmp_root, cls, 'images')
        stage_mask_dir = os.path.join(tmp_root, cls, 'masks')
        stage_out_dir = os.path.join(tmp_root, cls, 'output')
        
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(stage_img_dir, exist_ok=True)
        os.makedirs(stage_mask_dir, exist_ok=True)
        os.makedirs(stage_out_dir, exist_ok=True)
        
        all_files = glob.glob(os.path.join(in_dir, '*.png'))
        image_paths = [f for f in all_files if 'mask' not in os.path.basename(f)]
        gt_mask_paths = [f for f in all_files if 'mask' in os.path.basename(f)]

        # 1. Copy GT Masks directly to final output
        print(f"[{cls}] Copying GT masks...")
        for m in gt_mask_paths:
            shutil.copy2(m, os.path.join(out_dir, os.path.basename(m)))

        # 2. Stage images and generate masks with sanitized names
        print(f"[{cls}] Staging images and generating masks...")
        name_map = {} # sanitized -> original
        for img_path in image_paths:
            orig_name = os.path.basename(img_path)
            safe_name = sanitize_filename(orig_name)
            name_map[safe_name] = orig_name
            
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            
            # Save sanitized image
            cv2.imwrite(os.path.join(stage_img_dir, safe_name), img_bgr)
            # Generate and save sanitized mask
            mask = get_robust_mask_v2(img_bgr)
            cv2.imwrite(os.path.join(stage_mask_dir, safe_name), mask)
            
        # 3. Run LaMa on staged files
        print(f"[{cls}] Running LaMa batch inpainting (Staged)...")
        cmd = [
            "python", "-m", "iopaint", "run",
            "--model", "lama",
            "--image", stage_img_dir,
            "--mask", stage_mask_dir,
            "--output", stage_out_dir,
            "--device", "cuda"
        ]
        subprocess.run(cmd, check=True)
        
        # 4. Move files back with original names
        print(f"[{cls}] Restoring original names to output...")
        for safe_name, orig_name in name_map.items():
            processed_path = os.path.join(stage_out_dir, safe_name)
            if os.path.exists(processed_path):
                shutil.move(processed_path, os.path.join(out_dir, orig_name))
            else:
                # If iopaint skipped, copy original
                shutil.copy2(os.path.join(in_dir, orig_name), os.path.join(out_dir, orig_name))

    print("Success: BUSI dataset preprocessed with high-quality AI Inpainting (V2).")

finally:
    # Cleanup staging area
    # shutil.rmtree(tmp_root, ignore_errors=True)
    pass
