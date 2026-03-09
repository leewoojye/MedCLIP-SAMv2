import os

import torch
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from PIL import Image
from transformers import AutoProcessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_path = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/UDIAT2/udiat2_train_dpo_image2prompt.csv"
run_dir = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src/logs/biomedclip_dpo_udiat_v28"
model_path = os.path.join(run_dir, "best_params")
output_csv = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/visualize/rawdot_scores.csv"
output_plot = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/visualize/rawdot_plot.png"


def calculate_rawdot():
    print(f"Loading model from local checkpoint: {model_path}")

    import sys

    sys.path.append(os.getcwd())
    from biomed_model.modeling_biomed_clip import BiomedCLIPModel

    model = BiomedCLIPModel.from_pretrained(model_path, local_files_only=True).to(device)
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    print("Model/processor ready.")

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["filename", "filename_neg", "Caption", "Caption_neg"])

    rawdot_pos = []
    rawdot_neg = []
    logits_pos = []
    logits_neg = []

    print(f"Processing {len(df)} pairs...")
    with torch.no_grad():
        for idx, row in df.iterrows():
            img_pos = Image.open(row["filename"]).convert("RGB")
            img_neg = Image.open(row["filename_neg"]).convert("RGB")
            txt_pos = str(row["Caption"])
            txt_neg = str(row["Caption_neg"])

            try:
                pos_inputs = processor(
                    text=[txt_pos], images=img_pos, return_tensors="pt", padding=True, truncation=True
                )
                pos_inputs = {k: v.to(device) for k, v in pos_inputs.items()}
                outputs_pos = model(**pos_inputs)

                neg_inputs = processor(
                    text=[txt_neg], images=img_neg, return_tensors="pt", padding=True, truncation=True
                )
                neg_inputs = {k: v.to(device) for k, v in neg_inputs.items()}
                outputs_neg = model(**neg_inputs)

                # Raw dot between returned embeddings (no extra normalize in script).
                rawdot_pos.append(torch.sum(outputs_pos.image_embeds * outputs_pos.text_embeds, dim=-1).item())
                rawdot_neg.append(torch.sum(outputs_neg.image_embeds * outputs_neg.text_embeds, dim=-1).item())

                # CLIP logits for reference (scaled similarity).
                logits_pos.append(outputs_pos.logits_per_image.item())
                logits_neg.append(outputs_neg.logits_per_image.item())

                if (idx + 1) % 10 == 0:
                    print(
                        f"[{idx + 1}/{len(df)}] rawdot_pos={rawdot_pos[-1]:.4f}, "
                        f"rawdot_neg={rawdot_neg[-1]:.4f}, "
                        f"logit_pos={logits_pos[-1]:.4f}, logit_neg={logits_neg[-1]:.4f}"
                    )
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                rawdot_pos.append(None)
                rawdot_neg.append(None)
                logits_pos.append(None)
                logits_neg.append(None)

    df["rawdot_pos"] = rawdot_pos
    df["rawdot_neg"] = rawdot_neg
    df["logit_pos"] = logits_pos
    df["logit_neg"] = logits_neg
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

    plot_pos = [x for x in rawdot_pos if x is not None]
    plot_neg = [x for x in rawdot_neg if x is not None]

    plt.figure(figsize=(10, 6))
    if plot_pos:
        sns.histplot(plot_pos, color="blue", label="RawDot Positive", kde=True, stat="density", alpha=0.5)
    if plot_neg:
        sns.histplot(plot_neg, color="red", label="RawDot Negative", kde=True, stat="density", alpha=0.5)
    plt.title("Image/Text Raw Dot Distribution (UDIAT2, v28)")
    plt.xlabel("Raw dot of outputs.image_embeds and outputs.text_embeds")
    plt.ylabel("Density")
    if plot_pos or plot_neg:
        plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_plot)
    print(f"Saved: {output_plot}")


if __name__ == "__main__":
    calculate_rawdot()
