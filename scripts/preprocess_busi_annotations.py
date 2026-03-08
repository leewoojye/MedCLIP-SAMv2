import cv2
import glob
import os
import shutil
import numpy as np

input_base_dir = '/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/Dataset_BUSI_with_GT'
output_base_dir = '/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/BUSI_PREPROCESSED'
classes = ['benign', 'malignant', 'normal']

def remove_annotations_strict(img_bgr):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Very strict threshold: annotations are strictly added purely bright pixels (text/lines)
    # The user noted that lower thresholds catch tissue and darken it when inpainted.
    # So we strictly look for 245+
    _, thresh = cv2.threshold(img_gray, 245, 255, cv2.THRESH_BINARY)
    
    # Filter out anything that is large (because a large 255 area is likely dense calcification/tissue)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    
    mask = np.zeros_like(thresh)
    
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        
        # Annotations are small letters, dots, or thin lines.
        # If it's larger than 500 pixels in area, it's probably legitimate tissue.
        if area < 500: 
            mask[labels == i] = 255
            
    # Very slight dilation just to cover the anti-aliased edge of the text.
    # We do NOT want to expand the mask into surrounding tissue.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_dilated = cv2.dilate(mask, kernel, iterations=1)
    
    if cv2.countNonZero(mask_dilated) == 0:
        return img_bgr # Nothing to inpaint
        
    # We use cv2.inpaint instead of LaMa. 
    # LaMa "hallucinates" tissue which can darken or change the image globally.
    # cv2.inpaint just blurs/fills the exact 1-2 pixel width of the text, leaving the rest 100% untouched.
    inpainted = cv2.inpaint(img_bgr, mask_dilated, 3, cv2.INPAINT_TELEA)
    return inpainted

try:
    for cls in classes:
        in_dir = os.path.join(input_base_dir, cls)
        out_dir = os.path.join(output_base_dir, cls)
        os.makedirs(out_dir, exist_ok=True)
        
        images = glob.glob(os.path.join(in_dir, '*.png'))
        for img_path in images:
            basename = os.path.basename(img_path)
            out_path = os.path.join(out_dir, basename)
            
            if 'mask' in basename:
                shutil.copy2(img_path, out_path)
                continue
                
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
                
            processed_img = remove_annotations_strict(img_bgr)
            cv2.imwrite(out_path, processed_img)

    print("Strict preprocessing complete. Results saved in", output_base_dir)

except Exception as e:
    print(f"Error occurred: {e}")
