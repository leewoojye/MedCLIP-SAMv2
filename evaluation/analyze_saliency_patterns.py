"""
Saliency map과 예측 결과를 함께 분석하여 모델의 약점 파악
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

def analyze_with_saliency(case_id, base_dir):
    """
    Saliency map, GT, Prediction을 함께 분석
    """
    # 경로 설정
    image_path = base_dir / f'data/pancreas/test_images/pancreas_{case_id:03d}.png'
    gt_path = base_dir / f'data/pancreas/test_masks/pancreas_{case_id:03d}.png'
    pred_path = base_dir / f'sam_outputs/data/pancreas/test_masks/pancreas_{case_id:03d}.png'
    saliency_path = base_dir / f'saliency_map_outputs/data/pancreas/test_masks/pancreas_{case_id:03d}.png'
    
    # 이미지 로드
    try:
        image = np.array(Image.open(image_path).convert('L'))
        gt_mask = np.array(Image.open(gt_path).convert('L'))
        pred_mask = np.array(Image.open(pred_path).convert('L'))
        saliency = np.array(Image.open(saliency_path).convert('L'))
    except Exception as e:
        print(f"Error loading case {case_id}: {e}")
        return None
    
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
    
    # Saliency map 분석
    # GT 영역에서의 saliency
    gt_saliency = saliency[gt_bin > 0] if np.sum(gt_bin) > 0 else np.array([])
    # FP 영역에서의 saliency
    fp_saliency = saliency[fp_mask] if fp_count > 0 else np.array([])
    # FN 영역에서의 saliency
    fn_saliency = saliency[fn_mask] if fn_count > 0 else np.array([])
    
    # 시각화
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 첫 번째 행
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(saliency, cmap='hot')
    ax2.set_title(f'Saliency Map\n(mean={np.mean(saliency):.1f}, max={np.max(saliency)})')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(gt_mask, cmap='gray')
    ax3.set_title(f'Ground Truth\n(area={np.sum(gt_bin)} px)')
    ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(pred_mask, cmap='gray')
    ax4.set_title(f'Prediction\n(area={np.sum(pred_bin)} px, Dice={dice:.3f})')
    ax4.axis('off')
    
    # 두 번째 행 - Saliency 오버레이
    ax5 = fig.add_subplot(gs[1, 0])
    # Original + GT
    img_rgb = np.stack([image, image, image], axis=-1)
    overlay1 = img_rgb.copy().astype(np.float32)
    overlay1[gt_bin > 0] = overlay1[gt_bin > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    ax5.imshow(overlay1.astype(np.uint8))
    ax5.set_title('GT Overlay (Green)')
    ax5.axis('off')
    
    ax6 = fig.add_subplot(gs[1, 1])
    # Saliency + GT
    sal_rgb = plt.cm.hot(saliency / 255.0)[:, :, :3] * 255
    overlay2 = sal_rgb.copy().astype(np.float32)
    overlay2[gt_bin > 0] = overlay2[gt_bin > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
    ax6.imshow(overlay2.astype(np.uint8))
    ax6.set_title('Saliency + GT (Green)')
    ax6.axis('off')
    
    ax7 = fig.add_subplot(gs[1, 2])
    # Original + Prediction
    overlay3 = img_rgb.copy().astype(np.float32)
    overlay3[pred_bin > 0] = overlay3[pred_bin > 0] * 0.5 + np.array([255, 0, 255]) * 0.5
    ax7.imshow(overlay3.astype(np.uint8))
    ax7.set_title('Prediction Overlay (Magenta)')
    ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[1, 3])
    # FP/FN overlay
    overlay4 = img_rgb.copy().astype(np.float32)
    overlay4[fp_mask] = overlay4[fp_mask] * 0.4 + np.array([255, 0, 0]) * 0.6  # Red = FP
    overlay4[fn_mask] = overlay4[fn_mask] * 0.4 + np.array([0, 0, 255]) * 0.6  # Blue = FN
    overlay4[tp_mask] = overlay4[tp_mask] * 0.6 + np.array([0, 255, 0]) * 0.4  # Green = TP
    ax8.imshow(overlay4.astype(np.uint8))
    ax8.set_title('Error Analysis\n(Red=FP, Blue=FN, Green=TP)')
    ax8.axis('off')
    
    # 세 번째 행 - 통계
    ax9 = fig.add_subplot(gs[2, :2])
    
    # Saliency 히스토그램
    if len(gt_saliency) > 0:
        ax9.hist(gt_saliency, bins=50, alpha=0.7, label=f'GT area (mean={np.mean(gt_saliency):.1f})', 
                color='green', density=True)
    if len(fp_saliency) > 0:
        ax9.hist(fp_saliency, bins=50, alpha=0.7, label=f'FP area (mean={np.mean(fp_saliency):.1f})', 
                color='red', density=True)
    if len(fn_saliency) > 0:
        ax9.hist(fn_saliency, bins=50, alpha=0.7, label=f'FN area (mean={np.mean(fn_saliency):.1f})', 
                color='blue', density=True)
    
    ax9.set_xlabel('Saliency Value')
    ax9.set_ylabel('Density')
    ax9.set_title('Saliency Distribution by Region')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    ax10 = fig.add_subplot(gs[2, 2:])
    # 통계 텍스트
    stats_text = f"""
    케이스 분석: pancreas_{case_id:03d}
    
    Dice Score: {dice:.4f}
    
    영역 통계:
    - GT area: {np.sum(gt_bin)} 픽셀
    - Pred area: {np.sum(pred_bin)} 픽셀
    - TP: {tp_count}, FP: {fp_count}, FN: {fn_count}
    
    Saliency 통계:
    - GT 영역 평균: {np.mean(gt_saliency) if len(gt_saliency) > 0 else 0:.2f} ± {np.std(gt_saliency) if len(gt_saliency) > 0 else 0:.2f}
    - FP 영역 평균: {np.mean(fp_saliency) if len(fp_saliency) > 0 else 0:.2f} ± {np.std(fp_saliency) if len(fp_saliency) > 0 else 0:.2f}
    - FN 영역 평균: {np.mean(fn_saliency) if len(fn_saliency) > 0 else 0:.2f} ± {np.std(fn_saliency) if len(fn_saliency) > 0 else 0:.2f}
    - 전체 평균: {np.mean(saliency):.2f}
    
    문제 진단:
    """
    
    # 문제 진단
    if len(fp_saliency) > 0 and len(gt_saliency) > 0:
        if np.mean(fp_saliency) > np.mean(gt_saliency):
            stats_text += "\n    ⚠ FP 영역의 saliency가 GT보다 높음!"
            stats_text += "\n    → MedCLIP이 다른 부위를 췌장으로 강하게 인식"
        elif np.mean(fp_saliency) > np.mean(saliency) * 0.8:
            stats_text += "\n    ⚠ FP 영역에 높은 saliency 존재"
            stats_text += "\n    → 인접 장기와 췌장을 혼동"
    
    if len(fn_saliency) > 0 and len(gt_saliency) > 0:
        if np.mean(fn_saliency) < np.mean(gt_saliency) * 0.5:
            stats_text += "\n    ⚠ FN 영역의 saliency가 매우 낮음"
            stats_text += "\n    → 췌장 일부를 인식하지 못함"
    
    ax10.text(0.1, 0.5, stats_text, transform=ax10.transAxes, 
             fontsize=10, verticalalignment='center', family='monospace')
    ax10.axis('off')
    
    plt.suptitle(f'Pancreas Case {case_id:03d} - Comprehensive Analysis', 
                fontsize=16, fontweight='bold')
    
    # 저장
    output_dir = base_dir / 'evaluation/saliency_analysis'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f'pancreas_{case_id:03d}_full_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {output_path}")
    
    plt.close()
    
    # 분석 결과 반환
    return {
        'case_id': case_id,
        'dice': dice,
        'gt_saliency_mean': np.mean(gt_saliency) if len(gt_saliency) > 0 else 0,
        'fp_saliency_mean': np.mean(fp_saliency) if len(fp_saliency) > 0 else 0,
        'fn_saliency_mean': np.mean(fn_saliency) if len(fn_saliency) > 0 else 0,
        'overall_saliency_mean': np.mean(saliency),
    }

def main():
    base_dir = Path('/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2')
    
    # 주요 케이스 분석
    cases_to_analyze = [
        # 엄청난 과검출
        413, 386, 346, 5, 10,
        # 잘못된 위치
        4, 6, 12, 18,
        # 좋은 성능
        103, 297, 351
    ]
    
    results = []
    
    for case_id in cases_to_analyze:
        print(f"\n분석 중: Case {case_id:03d}")
        result = analyze_with_saliency(case_id, base_dir)
        if result:
            results.append(result)
    
    # 종합 분석
    print("\n" + "="*80)
    print("Saliency 기반 종합 분석")
    print("="*80)
    
    if results:
        # FP 영역의 saliency가 GT보다 높은 케이스
        fp_higher = [r for r in results if r['fp_saliency_mean'] > r['gt_saliency_mean']]
        print(f"\nFP 영역의 saliency가 GT보다 높은 케이스: {len(fp_higher)}/{len(results)}")
        for r in fp_higher:
            print(f"  Case {r['case_id']:03d}: FP={r['fp_saliency_mean']:.1f}, GT={r['gt_saliency_mean']:.1f}, "
                  f"Dice={r['dice']:.3f}")
        
        # FN 영역의 saliency가 낮은 케이스
        fn_low = [r for r in results if r['fn_saliency_mean'] < r['overall_saliency_mean'] * 0.5]
        print(f"\nFN 영역의 saliency가 전체 평균의 50% 미만인 케이스: {len(fn_low)}/{len(results)}")
        for r in fn_low:
            print(f"  Case {r['case_id']:03d}: FN={r['fn_saliency_mean']:.1f}, "
                  f"Overall={r['overall_saliency_mean']:.1f}")

if __name__ == '__main__':
    main()
