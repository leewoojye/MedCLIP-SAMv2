import pandas as pd
import os
import shutil

base_dir = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"
udiat3_dir = os.path.join(base_dir, "UDIAT3")
udiat4_dir = os.path.join(base_dir, "UDIAT4")

train_csv_path = os.path.join(udiat3_dir, "udiat_train_augmented.csv")
test_csv_path = os.path.join(udiat3_dir, "udiat_test_augmented.csv")

# Create folders
for sub in ["train_images", "train_masks", "test_images", "test_masks"]:
    os.makedirs(os.path.join(udiat4_dir, sub), exist_ok=True)

def get_mask_path(img_path):
    # /home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/UDIAT/Benign/000001.png
    # -> /home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/UDIAT/Benign_mask/000001.png
    if "UDIAT/Benign" in img_path:
        return img_path.replace("UDIAT/Benign", "UDIAT/Benign_mask")
    elif "UDIAT/Malignant" in img_path:
        return img_path.replace("UDIAT/Malignant", "UDIAT/Malignant_mask")
    return None

# Process Test
test_df = pd.read_csv(test_csv_path)
new_test_rows = []
for idx, row in test_df.iterrows():
    img_path = row['filename']
    img_name = os.path.basename(img_path)
    new_img_path = os.path.join(udiat4_dir, "test_images", img_name)
    
    # Copy Image
    if os.path.exists(img_path):
        shutil.copy2(img_path, new_img_path)
    
    # Copy Mask
    mask_path = get_mask_path(img_path)
    if mask_path and os.path.exists(mask_path):
        shutil.copy2(mask_path, os.path.join(udiat4_dir, "test_masks", img_name))
        
    row['filename'] = new_img_path
    new_test_rows.append(row)

new_test_df = pd.DataFrame(new_test_rows)
new_test_df.to_csv(os.path.join(udiat4_dir, "udiat_test_augmented.csv"), index=False)

# Process Train
train_df = pd.read_csv(train_csv_path)
new_train_rows = []
for idx, row in train_df.iterrows():
    # filename
    img_path = row['filename']
    img_name = os.path.basename(img_path)
    # If it's a negative image from closedform, it might have the same name as positive.
    # To avoid collision, we should distinguish them or use subfolders.
    # Let's check if names collide.
    if "generated_neg_output" in img_path:
        # It's a negative image. Let's name it neg_img_name
        dest_name = "neg_" + img_name
    else:
        dest_name = img_name
        
    new_img_path = os.path.join(udiat4_dir, "train_images", dest_name)
    if os.path.exists(img_path):
        shutil.copy2(img_path, new_img_path)
    
    # Mask
    mask_path = get_mask_path(img_path)
    if mask_path and os.path.exists(mask_path):
        shutil.copy2(mask_path, os.path.join(udiat4_dir, "train_masks", dest_name))
    
    row['filename'] = new_img_path
    
    # filename_neg
    neg_img_path = row['filename_neg']
    neg_img_name = os.path.basename(neg_img_path)
    if "generated_neg_output" in neg_img_path:
        neg_dest_name = "neg_" + neg_img_name
    else:
        neg_dest_name = neg_img_name
        
    new_neg_img_path = os.path.join(udiat4_dir, "train_images", neg_dest_name)
    if os.path.exists(neg_img_path):
        shutil.copy2(neg_img_path, new_neg_img_path)
        
    # Mask for neg (optional, but let's try)
    neg_mask_path = get_mask_path(neg_img_path)
    if neg_mask_path and os.path.exists(neg_mask_path):
        shutil.copy2(neg_mask_path, os.path.join(udiat4_dir, "train_masks", neg_dest_name))
        
    row['filename_neg'] = new_neg_img_path
    new_train_rows.append(row)

new_train_df = pd.DataFrame(new_train_rows)
new_train_df.to_csv(os.path.join(udiat4_dir, "udiat_train_augmented.csv"), index=False)

print(f"Organized UDIAT4 dataset with {len(new_test_df)} test rows and {len(new_train_df)} train rows.")
