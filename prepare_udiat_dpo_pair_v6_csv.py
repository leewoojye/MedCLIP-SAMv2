import pandas as pd
import os
import json

# Prompts (BUSI v6와 동일한 프롬프트 셋)
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

# Paths
base_dir = "/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"
# UDIAT original image sources
udiat_benign_dir = os.path.join(base_dir, "UDIAT/Benign")
udiat_malignant_dir = os.path.join(base_dir, "UDIAT/Malignant")

# Use pre-generated negative images instead of local noise
# udiat_clip_image2prompt_closedform (163 files)
neg_img_dir = os.path.join(base_dir, "generated_neg_output/udiat_clip_image2prompt_closedform")

# Prompt JSON mapping (163 files mapping)
json_path = os.path.join(base_dir, "saliency_maps/text_prompts/udiat2_test_prompts.json")
with open(json_path, 'r') as f:
    prompts_map = json.load(f)

print(f"Total entries in prompts_map: {len(prompts_map)}")

data = []
skipped = 0
processed = 0

for img_name in sorted(prompts_map.keys()):
    # 1. 원본 이미지 경로 찾기 (Benign 혹은 Malignant 폴더)
    img_path_benign = os.path.join(udiat_benign_dir, img_name)
    img_path_malignant = os.path.join(udiat_malignant_dir, img_name)
    
    if os.path.exists(img_path_benign):
        img_path = img_path_benign
        category = "benign"
        correct_prompts = benign_prompts
    elif os.path.exists(img_path_malignant):
        img_path = img_path_malignant
        category = "malignant"
        correct_prompts = malignant_prompts
    else:
        print(f"  [SKIP] {img_name}: original image not found in Benign/Malignant folders")
        skipped += 1
        continue

    # 2. 음성 이미지(neg_path) 경로 확인
    neg_path = os.path.join(neg_img_dir, img_name)
    if not os.path.exists(neg_path):
        print(f"  [SKIP] {img_name}: negative image file not found in {neg_img_dir}")
        skipped += 1
        continue

    # DPO 크로스-모달 페어 생성 (UDIAT v6 modified: Use image2prompt negatives)
    for p_correct, p_normal in zip(correct_prompts, normal_prompts):
        # 1. 원본 clean 이미지: 질병 캡션 선호 > 정상 캡션
        data.append({
            "filename": img_path,
            "Caption": p_correct,
            "filename_neg": img_path,
            "Caption_neg": p_normal
        })
        # 2. 음성 이미지 (image2prompt): 정상 캡션 선호 > 질병 캡션
        data.append({
            "filename": neg_path,
            "Caption": p_normal,
            "filename_neg": neg_path,
            "Caption_neg": p_correct
        })

    processed += 1

# CSV 저장
df_dpo = pd.DataFrame(data)
output_csv = os.path.join(
    base_dir,
    "biomedclip_finetuning/open_clip/src/data/breast_dataset/udiat_train_dpo_v6.csv"
)
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df_dpo.to_csv(output_csv, index=False)

print(f"\n{'='*50}")
print(f"Created UDIAT DPO v6 CSV (Using image2prompt Negatives)")
print(f"  Output  : {output_csv}")
print(f"  Processed: {processed} images")
print(f"  Skipped  : {skipped} images")
print(f"  Total rows: {len(df_dpo)}")
print(f"  (= {processed} images × 5 prompt pairs × 2 [original+negative] = {processed * 5 * 2} rows)")
