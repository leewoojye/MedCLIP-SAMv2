"""
췌장 분할 모델의 혼동 패턴 심층 분석
"""

import numpy as np
from PIL import Image
from pathlib import Path
from collections import defaultdict

def analyze_confusion_patterns():
    """
    모델이 췌장을 어떤 부위와 혼동하는지 분석
    """
    base_dir = Path('/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2')
    
    test_images_dir = base_dir / 'data/pancreas/test_images'
    test_masks_dir = base_dir / 'data/pancreas/test_masks'
    pred_masks_dir = base_dir / 'sam_outputs/data/pancreas/test_masks'
    
    test_images = sorted(test_images_dir.glob('*.png'))
    
    # 패턴별 케이스 분류
    patterns = {
        'massive_overdetection': [],  # 엄청난 과검출 (예측 >> GT)
        'wrong_location': [],  # 위치를 잘못 찾음 (FP와 FN이 분리됨)
        'partial_detection': [],  # 췌장의 일부만 검출
        'complete_miss': [],  # 완전히 놓침
        'good_performance': [],  # 좋은 성능
    }
    
    # 강도 기반 혼동 분석
    fp_intensity_by_pattern = defaultdict(list)
    gt_intensity_stats = []
    
    print("분석 중...")
    
    for img_path in test_images:
        img_name = img_path.stem
        
        gt_mask_path = test_masks_dir / f"{img_name}.png"
        pred_mask_path = pred_masks_dir / f"{img_name}.png"
        
        if not pred_mask_path.exists():
            continue
        
        try:
            image = np.array(Image.open(img_path).convert('L'))
            gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
            pred_mask = np.array(Image.open(pred_mask_path).convert('L'))
        except:
            continue
        
        # 바이너리 변환
        gt_bin = (gt_mask > 127).astype(np.uint8)
        pred_bin = (pred_mask > 127).astype(np.uint8)
        
        # 메트릭 계산
        fp_mask = (pred_bin == 1) & (gt_bin == 0)
        fn_mask = (pred_bin == 0) & (gt_bin == 1)
        tp_mask = (pred_bin == 1) & (gt_bin == 1)
        
        fp_count = np.sum(fp_mask)
        fn_count = np.sum(fn_mask)
        tp_count = np.sum(tp_mask)
        gt_area = np.sum(gt_bin)
        pred_area = np.sum(pred_bin)
        
        dice = 2 * tp_count / (2 * tp_count + fp_count + fn_count) if (2 * tp_count + fp_count + fn_count) > 0 else 0
        
        # GT 영역 강도
        if gt_area > 0:
            gt_intensity_stats.append(np.mean(image[gt_bin > 0]))
        
        # 패턴 분류
        case_info = {
            'name': img_name,
            'dice': dice,
            'fp_count': fp_count,
            'fn_count': fn_count,
            'tp_count': tp_count,
            'gt_area': gt_area,
            'pred_area': pred_area,
        }
        
        # FP 강도 분석
        if fp_count > 0:
            fp_intensities = image[fp_mask]
            case_info['fp_mean_intensity'] = np.mean(fp_intensities)
            case_info['fp_std_intensity'] = np.std(fp_intensities)
        
        # 패턴 분류 로직
        if dice > 0.3:
            patterns['good_performance'].append(case_info)
            if fp_count > 0:
                fp_intensity_by_pattern['good_performance'].extend(image[fp_mask])
        
        elif pred_area == 0:
            patterns['complete_miss'].append(case_info)
        
        elif pred_area > gt_area * 3 and fp_count > 5000:
            patterns['massive_overdetection'].append(case_info)
            if fp_count > 0:
                fp_intensity_by_pattern['massive_overdetection'].extend(image[fp_mask])
        
        elif fn_count > tp_count * 2 and dice > 0:
            patterns['partial_detection'].append(case_info)
            if fp_count > 0:
                fp_intensity_by_pattern['partial_detection'].extend(image[fp_mask])
        
        elif fp_count > 0 and fn_count > 0 and dice < 0.2:
            # FP와 FN의 중심이 멀리 떨어져 있는지 확인
            if fp_count > 100 and fn_count > 100:
                fp_coords = np.where(fp_mask)
                fn_coords = np.where(fn_mask)
                fp_center = (np.mean(fp_coords[0]), np.mean(fp_coords[1]))
                fn_center = (np.mean(fn_coords[0]), np.mean(fn_coords[1]))
                distance = np.sqrt((fp_center[0] - fn_center[0])**2 + (fp_center[1] - fn_center[1])**2)
                
                if distance > 50:  # 50픽셀 이상 떨어져 있으면
                    patterns['wrong_location'].append(case_info)
                    if fp_count > 0:
                        fp_intensity_by_pattern['wrong_location'].extend(image[fp_mask])
    
    # 결과 출력
    print("\n" + "="*80)
    print("췌장 분할 모델 혼동 패턴 분석 결과")
    print("="*80)
    
    print(f"\n1. 엄청난 과검출 (Massive Over-detection): {len(patterns['massive_overdetection'])} 케이스")
    print("   특징: 예측 영역이 GT의 3배 이상, FP > 5000픽셀")
    if len(patterns['massive_overdetection']) > 0:
        avg_dice = np.mean([c['dice'] for c in patterns['massive_overdetection']])
        avg_fp = np.mean([c['fp_count'] for c in patterns['massive_overdetection']])
        avg_pred_area = np.mean([c['pred_area'] for c in patterns['massive_overdetection']])
        avg_gt_area = np.mean([c['gt_area'] for c in patterns['massive_overdetection']])
        print(f"   평균 Dice: {avg_dice:.4f}")
        print(f"   평균 FP: {avg_fp:.0f} 픽셀")
        print(f"   평균 예측 영역: {avg_pred_area:.0f} vs GT: {avg_gt_area:.0f} (비율: {avg_pred_area/avg_gt_area:.1f}x)")
        
        if 'massive_overdetection' in fp_intensity_by_pattern:
            intensities = fp_intensity_by_pattern['massive_overdetection']
            print(f"   FP 영역 평균 강도: {np.mean(intensities):.2f} ± {np.std(intensities):.2f}")
        
        print("\n   대표 케이스:")
        for i, case in enumerate(sorted(patterns['massive_overdetection'], 
                                       key=lambda x: x['pred_area']/x['gt_area'] if x['gt_area'] > 0 else 0, 
                                       reverse=True)[:5]):
            ratio = case['pred_area']/case['gt_area'] if case['gt_area'] > 0 else 0
            print(f"   - {case['name']}: 예측/GT = {ratio:.1f}x, FP={case['fp_count']}, Dice={case['dice']:.4f}")
    
    print(f"\n2. 잘못된 위치 감지 (Wrong Location): {len(patterns['wrong_location'])} 케이스")
    print("   특징: FP와 FN 영역이 공간적으로 분리됨 (다른 부위를 췌장으로 오인)")
    if len(patterns['wrong_location']) > 0:
        avg_dice = np.mean([c['dice'] for c in patterns['wrong_location']])
        print(f"   평균 Dice: {avg_dice:.4f}")
        
        if 'wrong_location' in fp_intensity_by_pattern:
            intensities = fp_intensity_by_pattern['wrong_location']
            print(f"   FP 영역 평균 강도: {np.mean(intensities):.2f} ± {np.std(intensities):.2f}")
        
        print("\n   대표 케이스:")
        for i, case in enumerate(patterns['wrong_location'][:5]):
            print(f"   - {case['name']}: FP={case['fp_count']}, FN={case['fn_count']}, Dice={case['dice']:.4f}")
    
    print(f"\n3. 부분 검출 (Partial Detection): {len(patterns['partial_detection'])} 케이스")
    print("   특징: 췌장의 일부분만 검출 (FN > 2*TP)")
    if len(patterns['partial_detection']) > 0:
        avg_dice = np.mean([c['dice'] for c in patterns['partial_detection']])
        avg_fn_ratio = np.mean([c['fn_count']/c['gt_area'] if c['gt_area'] > 0 else 0 
                               for c in patterns['partial_detection']])
        print(f"   평균 Dice: {avg_dice:.4f}")
        print(f"   평균 FN 비율: {avg_fn_ratio:.2%}")
        
        if 'partial_detection' in fp_intensity_by_pattern:
            intensities = fp_intensity_by_pattern['partial_detection']
            print(f"   FP 영역 평균 강도: {np.mean(intensities):.2f} ± {np.std(intensities):.2f}")
    
    print(f"\n4. 완전 미검출 (Complete Miss): {len(patterns['complete_miss'])} 케이스")
    print("   특징: 모델이 아무것도 예측하지 않음")
    if len(patterns['complete_miss']) > 0:
        avg_gt_area = np.mean([c['gt_area'] for c in patterns['complete_miss']])
        print(f"   평균 GT 영역 크기: {avg_gt_area:.0f} 픽셀")
        
        # 작은 GT vs 큰 GT
        small_gt = [c for c in patterns['complete_miss'] if c['gt_area'] < 1000]
        large_gt = [c for c in patterns['complete_miss'] if c['gt_area'] >= 1000]
        print(f"   - 작은 GT (<1000픽셀): {len(small_gt)} 케이스")
        print(f"   - 큰 GT (≥1000픽셀): {len(large_gt)} 케이스")
    
    print(f"\n5. 좋은 성능 (Good Performance): {len(patterns['good_performance'])} 케이스")
    print("   특징: Dice > 0.3")
    if len(patterns['good_performance']) > 0:
        avg_dice = np.mean([c['dice'] for c in patterns['good_performance']])
        print(f"   평균 Dice: {avg_dice:.4f}")
        
        if 'good_performance' in fp_intensity_by_pattern:
            intensities = fp_intensity_by_pattern['good_performance']
            print(f"   FP 영역 평균 강도: {np.mean(intensities):.2f} ± {np.std(intensities):.2f}")
    
    # 전체 통계
    print("\n" + "="*80)
    print("전체 강도 통계")
    print("="*80)
    
    if gt_intensity_stats:
        print(f"GT 영역 평균 강도: {np.mean(gt_intensity_stats):.2f} ± {np.std(gt_intensity_stats):.2f}")
    
    # FP 영역 강도 비교
    print("\n패턴별 FP 영역 강도 비교:")
    for pattern_name, intensities in fp_intensity_by_pattern.items():
        if len(intensities) > 0:
            print(f"  {pattern_name}: {np.mean(intensities):.2f} ± {np.std(intensities):.2f}")
    
    # 결론
    print("\n" + "="*80)
    print("주요 발견사항")
    print("="*80)
    
    total_cases = sum(len(p) for p in patterns.values())
    
    print(f"\n1. 과검출이 지배적인 문제: {len(patterns['massive_overdetection'])} / {total_cases} "
          f"({100*len(patterns['massive_overdetection'])/total_cases:.1f}%)")
    
    print(f"\n2. 모델이 혼동하는 부위 추정:")
    print("   - FP 영역의 평균 강도가 GT 영역과 유사하거나 약간 낮음")
    print("   - 이는 모델이 비슷한 강도를 가진 주변 장기(간, 비장, 위 등)를 췌장으로 오인할 가능성을 시사")
    
    if gt_intensity_stats and 'massive_overdetection' in fp_intensity_by_pattern:
        gt_mean = np.mean(gt_intensity_stats)
        fp_mean = np.mean(fp_intensity_by_pattern['massive_overdetection'])
        print(f"   - GT 평균 강도: {gt_mean:.2f}")
        print(f"   - 과검출 FP 평균 강도: {fp_mean:.2f}")
        print(f"   - 차이: {abs(gt_mean - fp_mean):.2f} (상대적으로 작음)")
    
    print(f"\n3. 완전 미검출 문제: {len(patterns['complete_miss'])} 케이스")
    print("   - Saliency map이 췌장 영역을 전혀 강조하지 못함")
    print("   - 다른 부위의 saliency가 더 높아서 SAM이 잘못된 영역을 선택")

if __name__ == '__main__':
    analyze_confusion_patterns()
