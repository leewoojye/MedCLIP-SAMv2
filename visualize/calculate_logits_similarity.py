import os

import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoProcessor


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

csv_path = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/UDIAT2/udiat2_train_dpo_image2prompt.csv"
run_dir = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/biomedclip_finetuning/open_clip/src/logs/biomedclip_dpo_udiat_v28"
model_path = os.path.join(run_dir, "best_params")
output_csv = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/visualize/logits_scores.csv"
output_plot = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/visualize/logits_plot.png"


def calculate_logits_similarity():
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

    logit_pos = []
    logit_neg = []

    print(f"Processing {len(df)} pairs...")
    with torch.no_grad():
        for idx, row in df.iterrows():
            try:
                img_pos = Image.open(row["filename"]).convert("RGB")
                img_neg = Image.open(row["filename_neg"]).convert("RGB")
                txt_pos = str(row["Caption"])
                txt_neg = str(row["Caption_neg"])

                pos_inputs = processor(
                    text=[txt_pos],
                    images=img_pos,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                pos_inputs = {k: v.to(device) for k, v in pos_inputs.items()}
                outputs_pos = model(**pos_inputs)

                neg_inputs = processor(
                    text=[txt_neg],
                    images=img_neg,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                neg_inputs = {k: v.to(device) for k, v in neg_inputs.items()}
                outputs_neg = model(**neg_inputs)

                logit_pos_val = outputs_pos.logits_per_image.item()
                logit_neg_val = outputs_neg.logits_per_image.item()
                logit_pos.append(logit_pos_val)
                logit_neg.append(logit_neg_val)

                if (idx + 1) % 10 == 0:
                    print(
                        f"[{idx + 1}/{len(df)}] "
                        f"logit_pos={logit_pos_val:.4f}, logit_neg={logit_neg_val:.4f}"
                    )
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                logit_pos.append(None)
                logit_neg.append(None)

    df["logit_pos"] = logit_pos
    df["logit_neg"] = logit_neg
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

    plot_pos = [x for x in logit_pos if x is not None]
    plot_neg = [x for x in logit_neg if x is not None]

    # One sample = one point visualization.
    plot_df = pd.DataFrame(
        {
            "pair_type": (["Positive"] * len(plot_pos)) + (["Negative"] * len(plot_neg)),
            "logits_per_image": plot_pos + plot_neg,
        }
    )

    plt.figure(figsize=(10, 6))
    sns.stripplot(
        data=plot_df,
        x="pair_type",
        y="logits_per_image",
        hue="pair_type",
        jitter=0.25,
        dodge=False,
        alpha=0.65,
        size=4,
        palette={"Positive": "blue", "Negative": "red"},
        legend=False,
    )
    plt.title("Image/Text Logits (One Dot Per Sample, UDIAT2, v28)")
    plt.xlabel("Pair Type")
    plt.ylabel("logits_per_image")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(output_plot)
    print(f"Saved: {output_plot}")


if __name__ == "__main__":
    calculate_logits_similarity()
