"""
Phase 1: Extract paired embeddings from MedPix captions.
Pairs: Finetuned BiomedCLIP (DHN) text hidden states ↔ SD CLIP text hidden states.
Used to train a projection layer from BiomedCLIP space → SD CLIP space.
"""

import torch
import torch.nn.functional as F
import csv
import argparse
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, CLIPTokenizer, CLIPTextModel


def load_finetuned_biomedclip(model_path, device="cuda"):
    """Load MedCLIP-SAMv2 finetuned BiomedCLIP (DHN model)."""
    print(f"Loading Finetuned BiomedCLIP from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        "chuhac/BiomedCLIP-vit-bert-hf", trust_remote_code=True
    )
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device)
    if not hasattr(model.config.text_config, "is_decoder"):
        model.config.text_config.is_decoder = False
    model.eval()
    return tokenizer, model


def load_sd_clip(device="cuda"):
    """Load SD 1.5 CLIP text encoder (ViT-L/14)."""
    model_name = "openai/clip-vit-large-patch14"
    print(f"Loading SD CLIP from: {model_name}")
    tokenizer = CLIPTokenizer.from_pretrained(model_name)
    model = CLIPTextModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def get_biomedclip_hidden_states(tokenizer, model, text, device="cuda", target_len=77):
    """Get BiomedCLIP text encoder hidden states, aligned to target_len."""
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt")
    vocab_size = model.config.text_config.vocab_size

    if inputs["input_ids"].max() >= vocab_size:
        inputs["input_ids"] = torch.clamp(inputs["input_ids"], max=vocab_size - 1)

    inputs["token_type_ids"] = torch.zeros_like(inputs["input_ids"])
    seq_length = inputs["input_ids"].shape[1]
    batch_size = inputs["input_ids"].shape[0]
    inputs["position_ids"] = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        text_outputs = model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs["token_type_ids"],
            position_ids=inputs["position_ids"],
            output_hidden_states=True,
        )
        hidden = text_outputs[0]  # (batch, seq_len, 768)

    # Truncate or pad to target_len
    if hidden.shape[1] > target_len:
        hidden = hidden[:, :target_len, :]
    elif hidden.shape[1] < target_len:
        pad_size = target_len - hidden.shape[1]
        hidden = F.pad(hidden, (0, 0, 0, pad_size), "constant", 0)

    return hidden  # (batch, target_len, 768)


def get_sd_clip_hidden_states(tokenizer, model, text, device="cuda"):
    """Get SD CLIP text encoder hidden states."""
    inputs = tokenizer(
        text, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state  # (batch, 77, 768)

    return hidden


def load_medpix_captions(csv_path):
    """Load captions from MedPix CSV."""
    captions = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            caption = row.get("Caption", "").strip()
            if caption and len(caption) > 10:  # Filter very short captions
                captions.append(caption)
    return captions


def get_breast_anchors():
    """Generate breast-specific anchor texts (domain-general medical expressions)."""
    return [
        # Benign descriptions
        "A medical breast mammogram showing a well-defined, round mass suggestive of a benign breast tumor.",
        "A breast imaging scan showing a well-circumscribed, smooth mass indicative of a benign breast tumor.",
        "A mammogram displaying a smoothly contoured mass suggestive of a benign breast tumor.",
        "A breast mammogram identifying a well-encapsulated, smooth mass consistent with a benign tumor.",
        "A mammogram showing a sharply marginated, round mass suggestive of a benign breast tumor.",
        "A breast scan showing a well-contained, round mass likely indicating a benign breast tumor.",
        "Breast ultrasound revealing a well-defined oval hypoechoic mass with smooth margins.",
        "Sonographic evaluation of the breast showing a circumscribed solid lesion with benign features.",
        "Breast imaging demonstrating a round, well-marginated lesion consistent with fibroadenoma.",
        "Ultrasound of the breast depicting a homogeneous, well-encapsulated mass suggestive of a cyst.",
        # Malignant descriptions
        "A medical breast mammogram showing an irregularly shaped, spiculated mass suggestive of a malignant breast tumor.",
        "A mammogram displaying an irregular, non-homogeneous mass suggestive of a malignant breast tumor.",
        "A breast imaging study revealing an irregularly shaped, invasive mass likely representing a malignant breast tumor.",
        "A medical breast mammogram displaying a heterogeneous, irregular mass consistent with a malignant tumor.",
        "A breast scan identifying an ill-defined, spiculated mass highly indicative of a malignant breast tumor.",
        "Breast ultrasound showing an irregular hypoechoic mass with angular margins and posterior shadowing.",
        "Sonographic assessment revealing a taller-than-wide lesion with ill-defined borders in the breast.",
        "Breast imaging demonstrating architectural distortion with associated microcalcifications.",
        "Ultrasound of the breast depicting an irregular mass with internal vascularity on Doppler evaluation.",
        "Mammographic evaluation showing a dense irregular opacity with spiculated margins in the upper outer quadrant.",
        # Healthy / Normal descriptions
        "A breast mammogram displaying normal breast structures with no signs of disease or masses.",
        "A mammogram showing no abnormalities in the breast tissue, suggestive of a normal, healthy breast.",
        "A breast imaging scan displaying normal, well-distributed breast tissue with no evidence of masses or tumors.",
        "A breast imaging study showing clear, homogeneous breast tissue with no signs of pathology.",
        "A mammogram revealing normal breast tissue with no visible masses, calcifications, or distortions.",
        "A breast mammogram showing unremarkable findings consistent with normal, healthy breast tissue.",
        "A medical mammogram displaying well-organized breast tissue with no evidence of abnormalities.",
        "A medical breast mammogram showing healthy breast parenchyma with no suspicious findings.",
        "Normal breast ultrasound showing homogeneous fibroglandular tissue without focal lesions.",
        "Breast sonography demonstrating normal glandular tissue and fat echotexture bilaterally.",
        # Generic medical
        "tumor breast",
        "healthy breast",
        "benign breast mass",
        "malignant breast mass",
        "normal breast tissue",
        "breast ultrasound with tumor",
        "breast ultrasound without tumor",
        "healthy normal breast tissue without any masses or lesions",
        "breast imaging showing pathological findings",
        "breast imaging showing no pathological findings",
        # Cross-domain anchors for robustness
        "tumor brain",
        "healthy brain",
        "brain MRI showing a glioblastoma",
        "normal brain MRI without any abnormalities",
        "chest X-ray showing pneumonia",
        "normal chest X-ray without any abnormalities",
        "abdominal CT showing a liver mass",
        "normal abdominal CT without any lesions",
    ]


def main(args):
    device = args.device

    # 1. Load models
    bio_tokenizer, bio_model = load_finetuned_biomedclip(args.biomedclip_path, device)
    sd_tokenizer, sd_model = load_sd_clip(device)

    # 2. Load texts
    print(f"\nLoading MedPix captions from: {args.medpix_csv}")
    medpix_captions = load_medpix_captions(args.medpix_csv)
    breast_anchors = get_breast_anchors()

    all_texts = medpix_captions + breast_anchors
    print(f"Total texts: {len(all_texts)} (MedPix: {len(medpix_captions)}, Anchors: {len(breast_anchors)})")

    # 3. Extract paired embeddings
    print("\nExtracting paired embeddings...")
    bio_embeddings = []
    sd_embeddings = []
    valid_texts = []

    batch_size = args.batch_size
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Extracting"):
        batch_texts = all_texts[i : i + batch_size]

        try:
            bio_emb = get_biomedclip_hidden_states(
                bio_tokenizer, bio_model, batch_texts, device, target_len=77
            )
            sd_emb = get_sd_clip_hidden_states(sd_tokenizer, sd_model, batch_texts, device)

            bio_embeddings.append(bio_emb.cpu())
            sd_embeddings.append(sd_emb.cpu())
            valid_texts.extend(batch_texts)
        except Exception as e:
            print(f"Skipping batch {i}: {e}")
            continue

    # 4. Concatenate and save
    bio_all = torch.cat(bio_embeddings, dim=0)  # (N, 77, 768)
    sd_all = torch.cat(sd_embeddings, dim=0)  # (N, 77, 768)

    print(f"\nFinal shapes - BiomedCLIP: {bio_all.shape}, SD CLIP: {sd_all.shape}")
    print(f"Valid pairs: {len(valid_texts)}")

    output = {
        "bio_embeddings": bio_all,
        "sd_embeddings": sd_all,
        "texts": valid_texts,
        "num_medpix": len(medpix_captions),
        "num_anchors": len(breast_anchors),
    }

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    torch.save(output, args.output)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--biomedclip-path",
        type=str,
        default="./saliency_maps/model",
        help="Path to finetuned BiomedCLIP (DHN) model directory",
    )
    parser.add_argument(
        "--medpix-csv",
        type=str,
        default="./biomedclip_finetuning/open_clip/src/data/medpix_dataset/medpix_dataset_clean.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./zero_shot_translation/paired_embeddings_medpix.pt",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
