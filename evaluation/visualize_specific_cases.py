"""
특정 케이스의 이미지를 보고 시각적으로 분석
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def visualize_case(case_id, base_dir):
    """
    특정 케이스를 시각화
    """
    # 경로 설정
    image_path = base_dir / f'data/pancreas/test_images/pancreas_{case_id:03d}.png'
    gt_path = base_dir / f'data/pancreas/test_masks/pancreas_{case_id:03d}.png'
    pred_path = base_dir / f'sam_outputs/data/pancreas/test_masks/pancreas_{case_id:03d}.png'
    
    # 이미지 로드
    try:
        image = np.array(Image.open(image_path).convert('L'))
        gt_mask = np.array(Image.open(gt_path).convert('L'))
        pred_mask = np.array(Image.open(pred_path).convert('L'))
    except Exception as e:
        print(f"Error loading case {case_id}: {e}")
        return
    
    # 바이너리 마스크
    gt_bin = (gt_mask > 127).astype(np.uint8)
    pred_bin = (pred_mask > 127).astype(np.uint8)
    
    # FP, FN, TP
    fp_mask = (pred_bin == 1) & (gt_bin == 0)
    fn_mask = (pred_bin == 0) & (gt_bin == 1)
    tp_mask = (pred_bin == 1) & (gt_bin == 1)
    
    # Dice 계산
    tp_count = np.sum(tp_mask)
    fp_count = np.sum(fp_mask)
    fn_count = np.sum(fn_mask)
    dice = 2 * tp_count / (2 * tp_count + fp_count + fn_count) if (2 * tp_count + fp_count + fn_count) > 0 else 0
    
    # 오버레이 생성
    image_rgb = np.stack([image, image, image], axis=-1)
    overlay = image_rgb.astype(np.float32).copy()
    
    alpha = 0.6
    # FP: 빨간색
    overlay[fp_mask] = overlay[fp_mask] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    # FN: 파란색
    overlay[fn_mask] = overlay[fn_mask] * (1 - alpha) + np.array([0, 0, 255]) * alpha
    # TP: 초록색
    overlay[tp_mask] = overlay[tp_mask] * (1 - alpha) + np.array([0, 255, 0]) * alpha * 0.5
    
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    # 시각화
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(gt_mask, cmap='gray')
    axes[0, 1].set_title(f'Ground Truth (area={np.sum(gt_bin)})')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(pred_mask, cmap='gray')
    axes[0, 2].set_title(f'Prediction (area={np.sum(pred_bin)})')
    axes[0, 2].axis('off')
    
    axes[1, 0].imshow(overlay)
    axes[1, 0].set_title(f'Overlay (Dice={dice:.4f})\nRed=FP, Blue=FN, Green=TP')
    axes[1, 0].axis('off')
    
    # FP만
    fp_viz = np.zeros_like(image_rgb)
    fp_viz[fp_mask] = [255, 0, 0]
    axes[1, 1].imshow(fp_viz)
    axes[1, 1].set_title(f'False Positive Only (n={fp_count})')
    axes[1, 1].axis('off')
    
    # FN만
    fn_viz = np.zeros_like(image_rgb)
    fn_viz[fn_mask] = [0, 0, 255]
    axes[1, 2].imshow(fn_viz)
    axes[1, 2].set_title(f'False Negative Only (n={fn_count})')
    axes[1, 2].axis('off')
    
    plt.suptitle(f'Pancreas Case {case_id:03d}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 저장
    output_dir = base_dir / 'evaluation/analysis_viz'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'pancreas_{case_id:03d}_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")
    
    # FP 영역의 이미지 강도 분석
    if fp_count > 0:
        fp_intensities = image[fp_mask]
        print(f"\nCase {case_id} - FP 영역 분석:")
        print(f"  평균 강도: {np.mean(fp_intensities):.2f}")
        print(f"  강도 범위: [{np.min(fp_intensities)}, {np.max(fp_intensities)}]")
        print(f"  강도 표준편차: {np.std(fp_intensities):.2f}")
        
        # 전체 이미지와 비교
        print(f"\n  전체 이미지 평균 강도: {np.mean(image):.2f}")
        print(f"  GT 영역 평균 강도: {np.mean(image[gt_bin > 0]):.2f}" if np.sum(gt_bin) > 0 else "  GT 없음")
    
    plt.close()

def main():
    base_dir = Path('/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2')
    
    # 분석할 케이스들
    # 1. 완전 실패
    # 2. 심각한 과검출
    # 3. 중간 성능
    # 4. 좋은 성능
    
    cases_to_analyze = [
        # 완전 실패
        1, 4, 6, 15, 29,
        # 심각한 과검출
        5, 10, 16,
        # 중간 성능
        95, 100, 170, 204,
        # 좋은 성능
        103, 297, 351, 276, 318
    ]
    
    for case_id in cases_to_analyze:
        print(f"\n{'='*60}")
        print(f"Analyzing Case {case_id:03d}")
        print('='*60)
        visualize_case(case_id, base_dir)

if __name__ == '__main__':
    main()
