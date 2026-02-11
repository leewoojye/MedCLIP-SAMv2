"""
시각화 결과 샘플 이미지를 한 번에 보기 좋게 표시하는 데모 스크립트
"""
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 경로
viz_dir = Path("/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/evaluation/fp_fn_viz/breast_tumors_test")
comparison_dir = viz_dir / "comparison"

# 처음 6개 이미지 선택
viz_files = sorted(list(viz_dir.glob("*.png")))[:6]

fig, axes = plt.subplots(2, 6, figsize=(24, 8))

for idx, viz_file in enumerate(viz_files):
    # 오버레이 이미지
    img_overlay = cv2.imread(str(viz_file))
    img_overlay = cv2.cvtColor(img_overlay, cv2.COLOR_BGR2RGB)
    axes[0, idx].imshow(img_overlay)
    axes[0, idx].set_title(viz_file.name, fontsize=10)
    axes[0, idx].axis('off')
    
    # 비교 이미지
    comparison_file = comparison_dir / viz_file.name
    if comparison_file.exists():
        img_comp = cv2.imread(str(comparison_file))
        img_comp = cv2.cvtColor(img_comp, cv2.COLOR_BGR2RGB)
        axes[1, idx].imshow(img_comp)
        axes[1, idx].set_title(f"비교 - {viz_file.name}", fontsize=10)
        axes[1, idx].axis('off')

# 범례
fig.text(0.5, 0.02, '위: FP/FN 오버레이 (빨강=FP, 파랑=FN, 초록=TP) | 아래: 원본 | 예측 | GT 비교', 
         ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(str(viz_dir / "sample_visualization.png"), dpi=150, bbox_inches='tight')
print(f"✓ 샘플 시각화가 저장되었습니다: {viz_dir / 'sample_visualization.png'}")
plt.close()

print("\n생성된 파일:")
print(f"- 오버레이 이미지: {len(list(viz_dir.glob('*.png')))} 개")
print(f"- 비교 이미지: {len(list(comparison_dir.glob('*.png')))} 개")
