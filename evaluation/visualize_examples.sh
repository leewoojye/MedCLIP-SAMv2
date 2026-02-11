# #!/bin/bash

# # FP/FN 오버레이 시각화 예시
TIMESTAMP=$(date +"%Y-%m-%d_%H%M")

# # 예시 1: 뇌종양
# echo "뇌종양 예측 결과 시각화..."
# python evaluation/visualize_fp_fn.py \
#     --gt-path data/brain_tumors/test_masks \
#     --pred-path sam_outputs/data/brain_tumors/test_masks \
#     --image-path data/brain_tumors/test_images \
#     --output-path evaluation/fp_fn_viz/brain_tumors/${TIMESTAMP} \
#     --alpha 0.5 \
#     --comparison

# # 예시 2: 췌장암
# echo "췌장암 예측 결과 시각화..."
# python evaluation/visualize_fp_fn.py \
#     --gt-path data/pancreas/test_masks \
#     --pred-path sam_outputs/data/pancreas/test_masks \
#     --image-path data/pancreas/test_images \
#     --output-path evaluation/fp_fn_viz/pancreas/${TIMESTAMP} \
#     --alpha 0.5 \
#     --comparison

# # 예시 2: 용종
echo "용종 예측 결과 시각화..."
python evaluation/visualize_fp_fn.py \
    --gt-path data/polyp/test_masks \
    --pred-path sam_outputs/data/polyp/test_masks \
    --image-path data/polyp/test_images \
    --output-path evaluation/fp_fn_viz/polyp/${TIMESTAMP} \
    --alpha 0.5 \
    --comparison

# # 예시 2: 유방종양
# echo "유방종양 예측 결과 시각화..."
# python evaluation/visualize_fp_fn.py \
#     --gt-path data/breast_tumors/test_masks \
#     --pred-path sam_outputs/data/breast_tumors/test_masks \
#     --image-path data/breast_tumors/test_images \
#     --output-path evaluation/fp_fn_viz/breast_tumors/${TIMESTAMP} \
#     --alpha 0.5 \
#     --comparison

# # 예시 3: 서로 다른 단계의 결과 비교
# echo "Coarse 출력 vs SAM 출력 비교..."
# python evaluation/visualize_fp_fn.py \
#     --gt-path data/brain_tumors/test_masks \
#     --pred-path coarse_outputs/brain_tumors/masks \
#     --image-path data/brain_tumors/test_images \
#     --output-path evaluation/fp_fn_viz/coarse_brain_tumors \
#     --alpha 0.5

# # 예시 4: 투명도를 더 높게 (오버레이를 더 강하게)
# echo "높은 투명도로 시각화..."
# python evaluation/visualize_fp_fn.py \
#     --gt-path data/lung_CT/test_masks \
#     --pred-path sam_outputs/lung_CT/masks \
#     --image-path data/lung_CT/test_images \
#     --output-path evaluation/fp_fn_viz/lung_CT_high_alpha \
#     --alpha 0.7
