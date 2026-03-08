import pandas as pd
import os
import pathlib

def update_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    
    def map_neg_path(row):
        image_path = row['filename']
        basename = os.path.basename(image_path)
        
        # Check if it's UDIAT or BUSI based on path or filename
        if 'UDIAT/test' in image_path or 'UDIAT/train' in image_path or basename.startswith('udiat_') or (basename.isdigit() and len(basename.split('.')[0]) == 6):
            neg_dir = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/generated_neg_output/udiat_clip_image2prompt_closedform"
            return os.path.join(neg_dir, basename)
        else:
            # Assuming BUSI for others (has benign/malignant prefix usually)
            neg_dir = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/generated_neg_output/busi_clip_image2prompt_closedformv0"
            return os.path.join(neg_dir, basename)

    df['filename_neg'] = df.apply(map_neg_path, axis=1)
    df.to_csv(output_csv, index=False)
    print(f"Created {output_csv} with {len(df)} rows.")

if __name__ == "__main__":
    base_dir = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/UDIATnBUSI"
    
    # Train
    train_in = os.path.join(base_dir, "train_dpo.csv")
    train_out = os.path.join(base_dir, "udiatnbusi_train_dpo_image2prompt.csv")
    if os.path.exists(train_in):
        update_csv(train_in, train_out)
    else:
        print(f"Warning: {train_in} not found.")

    # Test
    test_in = os.path.join(base_dir, "test_dpo.csv")
    test_out = os.path.join(base_dir, "udiatnbusi_test_dpo_image2prompt.csv")
    if os.path.exists(test_in):
        update_csv(test_in, test_out)
    else:
        print(f"Warning: {test_in} not found.")
