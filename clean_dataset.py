import pandas as pd
import os
import argparse

def clean_csv(csv_path, output_path):
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Reading {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read CSV: {e}")
        return

    print(f"Original rows: {len(df)}")
    if 'filename' not in df.columns:
        print("Error: 'filename' column not found.")
        return

    valid_rows = []
    missing_count = 0
    
    # Check first row to infer path structure
    first_file = df.iloc[0]['filename']
    print(f"Sample filename: {first_file}")
    
    # If the user script sets CWD to project root, and filename is relative to it?
    # Or relative to the CSV?
    # The error was: FileNotFoundError: [Errno 2] No such file or directory: 'data/medpix_dataset/images/synpic100492.jpg'
    # This implies the code looks for 'data/medpix_dataset/images/synpic100492.jpg'
    # If the CSV is at 'data/medpix_dataset/medpix_dataset.csv'
    # And the filename in CSV is probably 'images/synpic100492.jpg' or just 'synpic100492.jpg' + separate logic.
    # The log said: Image.open(str(self.images[idx]))
    # CsvDataset implementation usually joins root?
    # OpenCLIP CsvDataset:
    # df = pd.read_csv(input_filename)
    # self.images = df[img_key].tolist()
    # It takes the path AS IS from the column.
    # If the column has 'data/medpix_dataset/images/synpic100492.jpg', it uses that.
    # So we check existence of that path relative to CWD.

    for idx, row in df.iterrows():
        fname = row['filename']
        if os.path.exists(fname):
            valid_rows.append(row)
        else:
            missing_count += 1
            if missing_count < 5:
                print(f"Missing: {fname}")

    print(f"Found {missing_count} missing files.")
    print(f"Remaining valid rows: {len(valid_rows)}")

    if len(valid_rows) == 0:
        print("Error: No valid images found. Check paths.")
        return

    new_df = pd.DataFrame(valid_rows)
    new_df.to_csv(output_path, index=False)
    print(f"Saved cleaned CSV to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="biomedclip_finetuning/open_clip/src/data/medpix_dataset/medpix_dataset.csv")
    parser.add_argument("--output", type=str, default="biomedclip_finetuning/open_clip/src/data/medpix_dataset/medpix_dataset_clean.csv")
    args = parser.parse_args()
    
    clean_csv(args.csv, args.output)
