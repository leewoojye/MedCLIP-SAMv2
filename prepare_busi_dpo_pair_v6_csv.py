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
# BUSI original image sources
busi_benign_dir = os.path.join(base_dir, "Dataset_BUSI_with_GT/benign")
busi_malignant_dir = os.path.join(base_dir, "Dataset_BUSI_with_GT/malignant")

# Use pre-generated negative images
neg_img_dir = os.path.join(base_dir, "generated_neg_output/busi_clip_image2prompt_closedformv1")

# Prompt JSON mapping
json_path = os.path.join(base_dir, "saliency_maps/text_prompts/busi_test_prompts.json")
with open(json_path, 'r') as f:
    prompts_map = json.load(f)

print(f"Total entries in prompts_map: {len(prompts_map)}")

data = []
skipped = 0
processed = 0

for full_img_name in sorted(prompts_map.keys()):
    # JSON key format: "category_filename" (e.g., "benign_benign (1).png")
    if "_" not in full_img_name:
        print(f"  [SKIP] {full_img_name}: unexpected filename format (missing '_')")
        skipped += 1
        continue
    
    parts = full_img_name.split("_")
    category_from_key = parts[0]
    img_name = "_".join(parts[1:])
    
    # 1. 원본 이미지 경로 찾기 (Benign 혹은 Malignant 폴더)
    img_path_benign = os.path.join(busi_benign_dir, img_name)
    img_path_malignant = os.path.join(busi_malignant_dir, img_name)
    
    if os.path.exists(img_path_benign):
        img_path = img_path_benign
        category = "benign"
        correct_prompts = benign_prompts
    elif os.path.exists(img_path_malignant):
        img_path = img_path_malignant
        category = "malignant"
        correct_prompts = malignant_prompts
    else:
        print(f"  [SKIP] {full_img_name}: original image not found in {busi_benign_dir} or {busi_malignant_dir}")
        skipped += 1
        continue

    # 2. 음성 이미지(neg_path) 경로 확인 (Negative images are named with the same JSON key)
    neg_path = os.path.join(neg_img_dir, full_img_name)
    if not os.path.exists(neg_path):
        print(f"  [SKIP] {full_img_name}: negative image file not found in {neg_img_dir}")
        skipped += 1
        continue

    # DPO 크로스-모달 페어 생성
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
output_dir = os.path.join(base_dir, "BUSI_augmented")
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, "busi_train_dpo_v6.csv")

df_dpo.to_csv(output_csv, index=False)

print(f"\n{'='*50}")
print(f"Created BUSI DPO v6 CSV (Using image2prompt Negatives)")
print(f"  Output  : {output_csv}")
print(f"  Processed: {processed} images")
print(f"  Skipped  : {skipped} images")
print(f"  Total rows: {len(df_dpo)}")
print(f"  (= {processed} images × 5 prompt pairs × 2 [original+negative] = {processed * 5 * 2} rows)")
