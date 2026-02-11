import pandas as pd
import os

csv_path = "data/medpix_dataset/medpix_dataset.csv"
output_path = "data/medpix_dataset/medpix_dataset_clean.csv"

# Check if file exists
if not os.path.exists(csv_path):
    print(f"CSV not found at {csv_path}")
    # Try looking for it, or maybe it was generated elsewhere?
    # User had: data/medpix_dataset/medpix_dataset.csv
    # If not found, create a dummy one or exit?
    # Based on log, it WAS found.
    exit(1)

df = pd.read_csv(csv_path)
print(f"Original rows: {len(df)}")

valid_rows = []
for idx, row in df.iterrows():
    # Attempt to construct path same as DataLoader
    # DataLoader failed on 'data/medpix_dataset/images/synpic100492.jpg'
    # Assuming 'filename' column has the relative path 'images/...'?
    # Or maybe 'filename' is just 'synpic100492.jpg' and base path is prepended?
    # Error path: data/medpix_dataset/images/synpic100492.jpg
    # If csv is at data/medpix_dataset/medpix_dataset.csv
    # And image is at data/medpix_dataset/images/...
    # Then checking file existence:
    
    # We'll check the path as it appeared in the error.
    # The error path seems to be relative to CWD.
    
    # Let's inspect the 'filename' column content first.
    pass

# We can't iterate efficiently without knowing exact structure inside CSV.
# Let's print first row.
print("First row:", df.iloc[0])
