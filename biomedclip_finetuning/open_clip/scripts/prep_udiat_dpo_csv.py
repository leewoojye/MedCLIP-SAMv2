import os
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate DPO CSV for UDIAT dataset")
    parser.add_argument("--pos_dir", type=str, default="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/UDIAT", help="Directory with positive (original) images")
    parser.add_argument("--neg_dir", type=str, default="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/generated_neg_output/udiat_vector_injection_breast2", help="Directory with negative (generated) images")
    parser.add_argument("--output_csv", type=str, default="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src/data/udiat_dpo.csv", help="Output CSV path")
    args = parser.parse_args()

    data = []

    # Categories in UDIAT
    categories = ["Benign", "Malignant"]

    for category in categories:
        pos_cat_dir = os.path.join(args.pos_dir, category)
        neg_cat_dir = os.path.join(args.neg_dir, category)

        if not os.path.exists(pos_cat_dir) or not os.path.exists(neg_cat_dir):
            print(f"Skipping {category} as directories do not exist.")
            continue

        valid_images = [f for f in os.listdir(pos_cat_dir) if f.endswith(".png")]

        for img_name in valid_images:
            pos_img_path = os.path.join(pos_cat_dir, img_name)
            neg_img_path = os.path.join(neg_cat_dir, img_name)

            # Check if negative image exists
            if not os.path.exists(neg_img_path):
                print(f"Warning: Negative image not found for {img_name} in {category}. Skipping.")
                continue

            # Append to data
            data.append({
                "filename": pos_img_path,
                "filename_neg": neg_img_path,
                # For Breast Ultrasound, we use the anchors that generated the images
                "Caption": f"a photo of a {category.lower()} tumor breast",
                "Caption_neg": "a photo of a healthy breast"
            })

    df = pd.DataFrame(data)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    df.to_csv(args.output_csv, index=False, sep=",")
    print(f"Successfully generated DPO CSV at {args.output_csv} with {len(df)} samples.")

if __name__ == "__main__":
    main()
