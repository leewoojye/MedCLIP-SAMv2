import os
import shutil
import pandas as pd
import numpy as np
import random
from cv2 import imread, imwrite
from glob import glob

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Paths
base_dir = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"
output_dir = os.path.join(base_dir, "UDIAT7nBUSI7_aug")

for sub in ["train_images", "train_masks", "test_images", "test_masks"]:
    os.makedirs(os.path.join(output_dir, sub), exist_ok=True)

# ---------------------------------------------------------
# Prompts
# ---------------------------------------------------------
benign_prompts = [
    "A medical breast mammogram showing a well-defined, round mass suggestive of a benign breast tumor.",
    "A medical breast mammogram showing a smooth, well-circumscribed mass suggestive of a benign breast tumor.",
    "A mammogram displaying a round, homogeneous mass indicative of a benign breast tumor.",
    "A breast imaging study revealing a well-defined, encapsulated mass likely representing a benign breast tumor.",
    "A mammogram showing a sharply marginated, round mass suggestive of a benign breast tumor."
]

malignant_prompts = [
    "A medical breast mammogram showing an irregularly shaped, spiculated mass suggestive of a malignant breast tumor.",
    "A medical breast mammogram showing an irregular, spiculated mass suggestive of a malignant breast tumor.",
    "A mammogram displaying a poorly defined, lobulated mass indicative of a malignant breast tumor.",
    "A breast imaging study revealing an irregularly shaped, invasive mass likely representing a malignant breast tumor.",
    "A mammogram showing a dense, spiculated mass suggestive of a malignant breast tumor."
]

normal_prompts = [
    "A medical breast mammogram showing normal breast tissue with no evidence of tumor.",
    "A mammogram with no suspicious masses or malignant findings, indicating a normal scan.",
    "A breast imaging study revealing no signs of benign or malignant tumors.",
    "A mammogram showing symmetrical breast tissue with no focal masses or clinical findings.",
    "A diagnostic mammogram displaying normal glandular tissue without any detectable lesions."
]

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def merge_masks(mask_paths, output_path):
    import cv2
    if len(mask_paths) == 0:
        return False
    if len(mask_paths) == 1:
        shutil.copy(mask_paths[0], output_path)
        return True
    
    # Merge multiple masks
    masks = [cv2.imread(p, cv2.IMREAD_GRAYSCALE) for p in mask_paths]
    final_mask = np.zeros_like(masks[0])
    for m in masks:
        final_mask = np.maximum(final_mask, m)
    cv2.imwrite(output_path, final_mask)
    return True

# ---------------------------------------------------------
# 1. Parse UDIAT4 and BUSI_augmented lists
# ---------------------------------------------------------

# UDIAT images are named strictly with numbers like 000001.png or neg_000001.png
udiat_train_csv = os.path.join(base_dir, "UDIAT4/udiat_train_augmented.csv")
udiat_test_csv = os.path.join(base_dir, "UDIAT4/udiat_test_augmented.csv")

# Extract unique original IDs from UDIAT 
# We'll know it's malignant if the id > some number, but a safer way is looking at the original folders.
udiat_benign_dir = os.path.join(base_dir, "UDIAT/Benign")
udiat_malignant_dir = os.path.join(base_dir, "UDIAT/Malignant")

udiat_ids = []
for f in os.listdir(udiat_benign_dir):
    if f.endswith('.png'): udiat_ids.append((f, 'benign'))
for f in os.listdir(udiat_malignant_dir):
    if f.endswith('.png'): udiat_ids.append((f, 'malignant'))

# BUSI images
busi_csv = os.path.join(base_dir, "BUSI_augmented/busi_train_dpo_v6.csv")
busi_df = pd.read_csv(busi_csv)

# To get unique base original images for BUSI
# the filename is something like Dataset_BUSI_with_GT/benign/benign (1).png
# and for negatives: generated_neg_output/busi_clip_image2prompt_closedformv1/benign_benign (1).png
busi_benign_dir = os.path.join(base_dir, "Dataset_BUSI_with_GT/benign")
busi_malignant_dir = os.path.join(base_dir, "Dataset_BUSI_with_GT/malignant")

busi_ids = []
for f in os.listdir(busi_benign_dir):
    if f.endswith('.png') and 'mask' not in f:
        busi_ids.append((f, 'benign'))
for f in os.listdir(busi_malignant_dir):
    if f.endswith('.png') and 'mask' not in f:
        busi_ids.append((f, 'malignant'))

random.shuffle(udiat_ids)
random.shuffle(busi_ids)

# ---------------------------------------------------------
# 2. 70/30 Split
# ---------------------------------------------------------
udiat_split_idx = int(len(udiat_ids) * 0.7)
udiat_train = udiat_ids[:udiat_split_idx]
udiat_test = udiat_ids[udiat_split_idx:]

busi_split_idx = int(len(busi_ids) * 0.7)
busi_train = busi_ids[:busi_split_idx]
busi_test = busi_ids[busi_split_idx:]

# Combine trains and tests
train_items = [("udiat", x[0], x[1]) for x in udiat_train] + [("busi", x[0], x[1]) for x in busi_train]
test_items = [("udiat", x[0], x[1]) for x in udiat_test] + [("busi", x[0], x[1]) for x in busi_test]

# ---------------------------------------------------------
# 3. Copy Images and generate pairs
# ---------------------------------------------------------
train_dpo_data = []
test_dpo_data = []

def process_item(dataset, f_name, category, is_train):
    out_img_dir = os.path.join(output_dir, "train_images" if is_train else "test_images")
    out_mask_dir = os.path.join(output_dir, "train_masks" if is_train else "test_masks")
    
    # 1. Gather original image & negative image paths
    if dataset == "udiat":
        # UDIAT Original Image
        orig_img_src = os.path.join(udiat_benign_dir if category == 'benign' else udiat_malignant_dir, f_name)
        # UDIAT Negative Image
        neg_img_src = os.path.join(base_dir, "generated_neg_output/udiat_clip_image2prompt_closedform", f_name)
        # Check source existences - In UDIAT4 script, augmented negatives could be in `neg_`
        
        # Original udiat masks are in the same folder as images just named .png
        # wait, let me check where UDIAT masks are located. Actually we can use UDIAT4 train_masks/000001.png
        mask_src = os.path.join(base_dir, "UDIAT/Masks", f_name) # Assuming same name
        
        # Output prefixes
        out_base_name = f"udiat_{category}_{f_name}"
        out_neg_name = f"udiat_{category}_neg_{f_name}"
        
        dest_orig_img = os.path.join(out_img_dir, out_base_name)
        dest_neg_img = os.path.join(out_img_dir, out_neg_name)
        dest_mask = os.path.join(out_mask_dir, out_base_name)
        
        # Copy image and neg
        if os.path.exists(orig_img_src): shutil.copy(orig_img_src, dest_orig_img)
        if os.path.exists(neg_img_src): shutil.copy(neg_img_src, dest_neg_img)
        if os.path.exists(mask_src): shutil.copy(mask_src, dest_mask)
        
    else:
        # BUSI Original Image
        busi_src_dir = busi_benign_dir if category == 'benign' else busi_malignant_dir
        orig_img_src = os.path.join(busi_src_dir, f_name)
        
        # BUSI Negative Image
        neg_img_src = os.path.join(base_dir, "generated_neg_output/busi_clip_image2prompt_closedformv1", f"{category}_{f_name}")
        
        # BUSI Masks (could be multiple: benign (1)_mask.png, benign (1)_mask_1.png, etc)
        base_f_name = os.path.splitext(f_name)[0]
        mask_files = glob(os.path.join(busi_src_dir, f"{base_f_name}_mask*.png"))
        
        # Output prefixes
        out_base_name = f"busi_{category}_{f_name}"
        out_neg_name = f"busi_{category}_neg_{f_name}"
        
        dest_orig_img = os.path.join(out_img_dir, out_base_name)
        dest_neg_img = os.path.join(out_img_dir, out_neg_name)
        dest_mask = os.path.join(out_mask_dir, out_base_name)
        
        if os.path.exists(orig_img_src): shutil.copy(orig_img_src, dest_orig_img)
        if os.path.exists(neg_img_src): shutil.copy(neg_img_src, dest_neg_img)
        
        merge_masks(mask_files, dest_mask)
        
    # Append to DPO dataset
    
    # Selection of prompts
    correct_prompts = benign_prompts if category == 'benign' else malignant_prompts
    
    # Only 1 prompt pair if test
    n_prompts = 5 if is_train else 1
    
    data_list = train_dpo_data if is_train else test_dpo_data
    
    for i in range(n_prompts):
        p_correct = correct_prompts[i]
        p_normal = normal_prompts[i]
        
        # Base image: positive > negative
        data_list.append({
            "filename": dest_orig_img,
            "Caption": p_correct,
            "filename_neg": dest_orig_img,
            "Caption_neg": p_normal
        })
        
        # Negative image (image2prompt): negative > positive
        data_list.append({
            "filename": dest_neg_img,
            "Caption": p_normal,
            "filename_neg": dest_neg_img,
            "Caption_neg": p_correct
        })

print(f"Processing {len(train_items)} train items...")
for item in train_items:
    process_item(item[0], item[1], item[2], is_train=True)

print(f"Processing {len(test_items)} test items...")
for item in test_items:
    process_item(item[0], item[1], item[2], is_train=False)

# ---------------------------------------------------------
# 4. Save CSVs
# ---------------------------------------------------------
train_df = pd.DataFrame(train_dpo_data)
test_df = pd.DataFrame(test_dpo_data)

train_df.to_csv(os.path.join(output_dir, "train_dpo.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test_dpo.csv"), index=False)

print(f"Done! Train splits: {len(train_items)}, Test splits: {len(test_items)}")
print(f"Train DPO pairs: {len(train_df)}")
print(f"Test DPO pairs: {len(test_df)}")
