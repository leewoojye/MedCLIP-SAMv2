
import torch
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoProcessor
import torch.nn.functional as F

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
csv_path = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/UDIAT2/udiat2_train_dpo_image2prompt.csv"
run_dir = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src/logs/biomedclip_dpo_udiat_v28"
model_path = os.path.join(run_dir, "best_params")
output_csv = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/visualize/similarity_scores.csv"
output_plot = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/visualize/similarity_plot.png"

def calculate_similarity():
    # Load model and processor
    print(f"Loading model from local checkpoint: {model_path}")
    # Add current dir to path to find biomed_model
    import sys
    sys.path.append(os.getcwd())
    
    try:
        from biomed_model.modeling_biomed_clip import BiomedCLIPModel

        # Always use local fine-tuned weights for similarity calculation.
        model = BiomedCLIPModel.from_pretrained(model_path, local_files_only=True).to(device)
        print("Model loaded from local checkpoint.")
    except Exception as e:
        raise RuntimeError(f"Failed to load local model from {model_path}: {e}") from e

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    print("Processor loaded from local checkpoint directory.")

    model.eval()

    # Read CSV
    df = pd.read_csv(csv_path)
    # Remove empty lines if any
    df = df.dropna(subset=['filename', 'filename_neg', 'Caption', 'Caption_neg'])
    
    pos_similarities = []
    neg_similarities = []

    print(f"Processing {len(df)} pairs...")
    with torch.no_grad():
        for idx, row in df.iterrows():
            if idx < 3:
                print(f"Processing Row {idx}: {row['filename']}")
            
            # Positive pair
            img_pos_path = row['filename']
            txt_pos = str(row['Caption'])
            
            # Negative pair
            img_neg_path = row['filename_neg']
            txt_neg = str(row['Caption_neg'])
            
            try:
                # Process positive pair
                img_pos = Image.open(img_pos_path).convert("RGB")
                pos_inputs = processor(
                    text=[txt_pos],
                    images=img_pos,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                pos_inputs = {k: v.to(device) for k, v in pos_inputs.items()}
                outputs_pos = model(**pos_inputs)
                
                # Normalize features
                img_features_pos = F.normalize(outputs_pos.image_embeds, p=2, dim=-1)
                txt_features_pos = F.normalize(outputs_pos.text_embeds, p=2, dim=-1)
                
                # Cosine similarity
                sim_pos = torch.sum(img_features_pos * txt_features_pos, dim=-1).item()
                pos_similarities.append(sim_pos)
                
                # Process negative pair
                img_neg = Image.open(img_neg_path).convert("RGB")
                neg_inputs = processor(
                    text=[txt_neg],
                    images=img_neg,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                neg_inputs = {k: v.to(device) for k, v in neg_inputs.items()}
                outputs_neg = model(**neg_inputs)
                
                img_features_neg = F.normalize(outputs_neg.image_embeds, p=2, dim=-1)
                txt_features_neg = F.normalize(outputs_neg.text_embeds, p=2, dim=-1)
                
                sim_neg = torch.sum(img_features_neg * txt_features_neg, dim=-1).item()
                neg_similarities.append(sim_neg)
                
                if (idx + 1) % 10 == 0:
                    print(f"[{idx + 1}/{len(df)}] Completed... Current Sim: Pos={sim_pos:.4f}, Neg={sim_neg:.4f}")
                    
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                pos_similarities.append(None)
                neg_similarities.append(None)

    # Add to dataframe and save
    df['sim_pos'] = pos_similarities
    df['sim_neg'] = neg_similarities
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    # Visualization
    plt.figure(figsize=(10, 6))
    
    # Remove None values for plotting
    plot_pos = [s for s in pos_similarities if s is not None]
    plot_neg = [s for s in neg_similarities if s is not None]
    
    if plot_pos:
        sns.histplot(plot_pos, color="blue", label="Positive Pairs", kde=True, stat="density", alpha=0.5)
    if plot_neg:
        sns.histplot(plot_neg, color="red", label="Negative Pairs", kde=True, stat="density", alpha=0.5)
    
    plt.title("BioMedCLIP Image-Text Similarity Distribution (UDIAT2)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Density")
    if plot_pos or plot_neg:
        plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    plt.show()

if __name__ == "__main__":
    calculate_similarity()
