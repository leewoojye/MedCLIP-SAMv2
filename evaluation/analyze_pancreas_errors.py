"""
췌장 분할 결과 분석: FP/FN 패턴 분석
"""

import numpy as np
from PIL import Image
import os
from pathlib import Path
from collections import defaultdict

def analyze_image_errors(image_path, pred_mask_path, gt_mask_path):
    """
    개별 이미지의 에러 패턴 분석
    """
    # 이미지 로드
    try:
        image = np.array(Image.open(image_path).convert('L'))
        pred_mask = np.array(Image.open(pred_mask_path).convert('L'))
        gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
    except:
        return None
    
    # 바이너리 마스크로 변환
    pred_bin = (pred_mask > 127).astype(np.uint8)
    gt_bin = (gt_mask > 127).astype(np.uint8)
    
    # FP, FN, TP 계산
    fp_mask = (pred_bin == 1) & (gt_bin == 0)
    fn_mask = (pred_bin == 0) & (gt_bin == 1)
    tp_mask = (pred_bin == 1) & (gt_bin == 1)
    
    fp_count = np.sum(fp_mask)
    fn_count = np.sum(fn_mask)
    tp_count = np.sum(tp_mask)
    
    gt_area = np.sum(gt_bin)
    pred_area = np.sum(pred_bin)
    
    # 위치 분석
    if fp_count > 0:
        fp_coords = np.where(fp_mask)
        fp_center = (np.mean(fp_coords[1]), np.mean(fp_coords[0]))
        fp_regions = analyze_spatial_distribution(fp_mask)
    else:
        fp_center = None
        fp_regions = None
    
    if fn_count > 0:
        fn_coords = np.where(fn_mask)
        fn_center = (np.mean(fn_coords[1]), np.mean(fn_coords[0]))
        fn_regions = analyze_spatial_distribution(fn_mask)
    else:
        fn_center = None
        fn_regions = None
    
    # Dice 계산
    dice = 2 * tp_count / (2 * tp_count + fp_count + fn_count) if (2 * tp_count + fp_count + fn_count) > 0 else 0
    
    # 인접 장기 분석 (이미지 강도 기반)
    fp_intensity_stats = None
    if fp_count > 0 and fp_count < 1000000:  # 너무 많지 않은 경우만
        fp_intensities = image[fp_mask]
        fp_intensity_stats = {
            'mean': np.mean(fp_intensities),
            'std': np.std(fp_intensities),
            'min': np.min(fp_intensities),
            'max': np.max(fp_intensities)
        }
    
    return {
        'fp_count': fp_count,
        'fn_count': fn_count,
        'tp_count': tp_count,
        'gt_area': gt_area,
        'pred_area': pred_area,
        'dice': dice,
        'fp_center': fp_center,
        'fn_center': fn_center,
        'fp_regions': fp_regions,
        'fn_regions': fn_regions,
        'fp_intensity_stats': fp_intensity_stats,
        'fp_ratio': fp_count / pred_area if pred_area > 0 else 0,
        'fn_ratio': fn_count / gt_area if gt_area > 0 else 0,
    }

def analyze_spatial_distribution(mask):
    """
    마스크의 공간적 분포 분석 (이미지를 4구역으로 나눔)
    """
    h, w = mask.shape
    regions = {
        'upper_left': np.sum(mask[:h//2, :w//2]),
        'upper_right': np.sum(mask[:h//2, w//2:]),
        'lower_left': np.sum(mask[h//2:, :w//2]),
        'lower_right': np.sum(mask[h//2:, w//2:])
    }
    return regions

def main():
    base_dir = Path('/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2')
    
    # 데이터 경로
    test_images_dir = base_dir / 'data/pancreas/test_images'
    test_masks_dir = base_dir / 'data/pancreas/test_masks'
    pred_masks_dir = base_dir / 'sam_outputs/data/pancreas/test_masks'
    
    # 모든 테스트 이미지 찾기
    test_images = sorted(test_images_dir.glob('*.png'))
    
    results = []
    
    print(f"총 {len(test_images)}개의 테스트 이미지 분석 중...")
    
    for img_path in test_images:
        img_name = img_path.stem
        
        # 대응하는 마스크 찾기
        gt_mask_path = test_masks_dir / f"{img_name}.png"
        pred_mask_path = pred_masks_dir / f"{img_name}.png"
        
        if not pred_mask_path.exists():
            print(f"예측 마스크 없음: {img_name}")
            continue
        
        result = analyze_image_errors(str(img_path), str(pred_mask_path), str(gt_mask_path))
        
        if result:
            result['image_name'] = img_name
            results.append(result)
    
    # 통계 분석
    print("\n=== 전체 통계 ===")
    total_cases = len(results)
    print(f"분석된 케이스: {total_cases}")
    
    # Dice score 분포
    dice_scores = [r['dice'] for r in results]
    print(f"\nDice Score 통계:")
    print(f"  평균: {np.mean(dice_scores):.4f}")
    print(f"  중앙값: {np.median(dice_scores):.4f}")
    print(f"  표준편차: {np.std(dice_scores):.4f}")
    print(f"  최소: {np.min(dice_scores):.4f}")
    print(f"  최대: {np.max(dice_scores):.4f}")
    
    # 실패 케이스 (Dice < 0.1)
    failed_cases = [r for r in results if r['dice'] < 0.1]
    print(f"\n심각한 실패 케이스 (Dice < 0.1): {len(failed_cases)} / {total_cases} ({100*len(failed_cases)/total_cases:.1f}%)")
    
    # 완전 실패 (Dice = 0)
    complete_failures = [r for r in results if r['dice'] == 0]
    print(f"완전 실패 (Dice = 0): {len(complete_failures)} / {total_cases} ({100*len(complete_failures)/total_cases:.1f}%)")
    
    # FP vs FN 분석
    print("\n=== FP vs FN 분석 ===")
    fp_dominant = [r for r in results if r['fp_count'] > r['fn_count'] and r['dice'] < 0.5]
    fn_dominant = [r for r in results if r['fn_count'] > r['fp_count'] and r['dice'] < 0.5]
    balanced_error = [r for r in results if abs(r['fp_count'] - r['fn_count']) < 100 and r['dice'] < 0.5]
    
    print(f"FP 우세 (과검출): {len(fp_dominant)} 케이스")
    print(f"FN 우세 (미검출): {len(fn_dominant)} 케이스")
    print(f"균형적 에러: {len(balanced_error)} 케이스")
    
    # 미검출이 심각한 케이스 (FN이 GT의 80% 이상)
    severe_underdetection = [r for r in results if r['fn_ratio'] > 0.8]
    print(f"\n심각한 미검출 (FN > GT의 80%): {len(severe_underdetection)} 케이스")
    
    # 과검출이 심각한 케이스
    severe_overdetection = [r for r in results if r['pred_area'] > r['gt_area'] * 2 and r['fp_count'] > 1000]
    print(f"심각한 과검출 (예측 > GT의 2배): {len(severe_overdetection)} 케이스")
    
    # FP 강도 분석 (오검출된 영역의 이미지 강도)
    print("\n=== False Positive 영역 이미지 강도 분석 ===")
    fp_intensity_results = [r for r in results if r['fp_intensity_stats'] is not None]
    
    if fp_intensity_results:
        mean_intensities = [r['fp_intensity_stats']['mean'] for r in fp_intensity_results]
        print(f"FP 영역 평균 강도: {np.mean(mean_intensities):.2f} ± {np.std(mean_intensities):.2f}")
        print(f"FP 영역 강도 범위: [{np.min([r['fp_intensity_stats']['min'] for r in fp_intensity_results])}, "
              f"{np.max([r['fp_intensity_stats']['max'] for r in fp_intensity_results])}]")
    
    # 예시 케이스 출력
    print("\n=== 대표 실패 케이스 ===")
    
    # 최악의 케이스
    worst_cases = sorted(results, key=lambda x: x['dice'])[:10]
    print("\n최악의 10개 케이스:")
    for i, case in enumerate(worst_cases, 1):
        print(f"{i}. {case['image_name']}: Dice={case['dice']:.4f}, "
              f"FP={case['fp_count']}, FN={case['fn_count']}, GT_area={case['gt_area']}")
    
    # 최고 성능 케이스
    best_cases = sorted(results, key=lambda x: x['dice'], reverse=True)[:5]
    print("\n최고 성능 5개 케이스:")
    for i, case in enumerate(best_cases, 1):
        print(f"{i}. {case['image_name']}: Dice={case['dice']:.4f}, "
              f"FP={case['fp_count']}, FN={case['fn_count']}")
    
    # 패턴 분석
    print("\n=== 에러 패턴 요약 ===")
    
    # 완전 미검출 케이스
    no_detection = [r for r in results if r['pred_area'] == 0]
    print(f"\n1. 완전 미검출 (예측 없음): {len(no_detection)} 케이스")
    if len(no_detection) > 0:
        avg_gt_size = np.mean([r['gt_area'] for r in no_detection])
        print(f"   평균 GT 크기: {avg_gt_size:.0f} 픽셀")
    
    # 부분 검출 케이스
    partial_detection = [r for r in results if 0 < r['dice'] < 0.3 and r['pred_area'] > 0]
    print(f"\n2. 부분 검출 (0 < Dice < 0.3): {len(partial_detection)} 케이스")
    if len(partial_detection) > 0:
        avg_fn_ratio = np.mean([r['fn_ratio'] for r in partial_detection])
        print(f"   평균 FN 비율: {avg_fn_ratio:.2%}")
    
    # 과검출 케이스
    over_detection = [r for r in results if r['fp_count'] > r['fn_count'] * 2 and r['fp_count'] > 1000]
    print(f"\n3. 과검출 우세 (FP > 2*FN): {len(over_detection)} 케이스")
    if len(over_detection) > 0:
        for case in over_detection[:5]:
            print(f"   - {case['image_name']}: FP={case['fp_count']}, FN={case['fn_count']}, Dice={case['dice']:.4f}")

if __name__ == '__main__':
    main()
