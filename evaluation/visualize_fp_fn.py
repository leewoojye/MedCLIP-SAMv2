"""
FP(False Positive)와 FN(False Negative)를 색깔별로 오버레이하여 시각화하는 스크립트
- FP (False Positive): 빨간색 - 모델이 예측했지만 GT에는 없음
- FN (False Negative): 파란색 - GT에는 있지만 모델이 예측하지 못함  
- TP (True Positive): 초록색 - 모델과 GT 모두 일치
- 원본 이미지: 배경으로 표시
"""

import numpy as np
import cv2
import os
from pathlib import Path
import argparse
from tqdm import tqdm

join = os.path.join
basename = os.path.basename


def create_error_visualization(image, pred_mask, gt_mask, alpha=0.5):
    """
    예측 결과와 GT를 비교하여 FP/FN을 색깔별로 오버레이
    
    Args:
        image: 원본 의료 이미지 (HxWx3 또는 HxW)
        pred_mask: 모델 예측 마스크 (HxW, binary)
        gt_mask: Ground Truth 마스크 (HxW, binary)
        alpha: 오버레이 투명도 (0.0 ~ 1.0)
    
    Returns:
        오버레이된 이미지
    """
    # 이미지 정규화 및 BGR로 변환
    if len(image.shape) == 2:  # Grayscale
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_rgb = image.copy()
    
    # 마스크가 0-255 범위인 경우 0-1로 정규화
    pred_mask = pred_mask.astype(np.float32) / 255.0 if pred_mask.max() > 1 else pred_mask.astype(np.float32)
    gt_mask = gt_mask.astype(np.float32) / 255.0 if gt_mask.max() > 1 else gt_mask.astype(np.float32)
    
    # FP, FN, TP 계산
    fp_mask = (pred_mask > 0.5) & (gt_mask < 0.5)  # 모델이 1, GT가 0
    fn_mask = (pred_mask < 0.5) & (gt_mask > 0.5)  # 모델이 0, GT가 1
    tp_mask = (pred_mask > 0.5) & (gt_mask > 0.5)  # 모델과 GT 모두 1
    
    # 오버레이 이미지 생성
    overlay = image_rgb.astype(np.float32).copy()
    
    # FP: 빨간색
    overlay[fp_mask] = overlay[fp_mask] * (1 - alpha) + np.array([0, 0, 255]) * alpha
    
    # FN: 파란색
    overlay[fn_mask] = overlay[fn_mask] * (1 - alpha) + np.array([255, 0, 0]) * alpha
    
    # TP: 초록색 (선택사항)
    overlay[tp_mask] = overlay[tp_mask] * (1 - alpha) + np.array([0, 255, 0]) * alpha * 0.3
    
    return np.uint8(overlay), fp_mask, fn_mask, tp_mask


def create_comparison_image(image, pred_mask, gt_mask):
    """
    원본, 예측, GT를 나란히 보여주는 이미지 생성
    """
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_rgb = image.copy()
    
    # 마스크를 컬러로 변환
    pred_color = cv2.cvtColor((pred_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    gt_color = cv2.cvtColor((gt_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # 가로로 연결
    comparison = np.hstack([image_rgb, pred_color, gt_color])
    
    return comparison


def compute_metrics(pred_mask, gt_mask):
    """
    FP, FN, TP, TN 개수 및 비율 계산
    """
    pred_mask = (pred_mask > 127).astype(np.int32)
    gt_mask = (gt_mask > 127).astype(np.int32)
    
    tp = np.sum((pred_mask == 1) & (gt_mask == 1))
    fp = np.sum((pred_mask == 1) & (gt_mask == 0))
    fn = np.sum((pred_mask == 0) & (gt_mask == 1))
    tn = np.sum((pred_mask == 0) & (gt_mask == 0))
    
    total = tp + fp + fn + tn
    
    metrics = {
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'Total': total,
    }
    
    if (tp + fp) > 0:
        metrics['Precision'] = tp / (tp + fp)
    else:
        metrics['Precision'] = 0.0
    
    if (tp + fn) > 0:
        metrics['Recall'] = tp / (tp + fn)
    else:
        metrics['Recall'] = 0.0
    
    if (metrics['Precision'] + metrics['Recall']) > 0:
        metrics['F1'] = 2 * (metrics['Precision'] * metrics['Recall']) / (metrics['Precision'] + metrics['Recall'])
    else:
        metrics['F1'] = 0.0

    # Dice 계수는 바이너리 분할에서 F1과 동일
    denominator = (2 * tp + fp + fn)
    metrics['Dice'] = (2 * tp) / denominator if denominator > 0 else 0.0
    
    return metrics


def visualize_results(gt_path, pred_path, image_path, output_path, alpha=0.5, create_comparison=False):
    """
    모든 이미지에 대해 FP/FN 시각화 수행
    
    Args:
        gt_path: GT 마스크 디렉토리
        pred_path: 예측 마스크 디렉토리
        image_path: 원본 이미지 디렉토리
        output_path: 결과 저장 디렉토리
        alpha: 오버레이 투명도
        create_comparison: 비교 이미지 생성 여부
    """
    # 출력 디렉토리 생성
    Path(output_path).mkdir(parents=True, exist_ok=True)
    comparison_dir = join(output_path, 'comparison') if create_comparison else None
    
    if create_comparison:
        Path(comparison_dir).mkdir(parents=True, exist_ok=True)
    
    # 마스크 파일 목록 가져오기
    mask_files = os.listdir(pred_path)
    mask_files = [f for f in mask_files if f.endswith(('.png', '.jpg', '.jpeg'))]
    mask_files.sort()
    
    print(f"총 {len(mask_files)}개의 이미지를 처리합니다.\n")
    
    all_metrics = []
    
    for filename in tqdm(mask_files, desc="Processing"):
        # 파일 경로
        gt_file = join(gt_path, filename)
        pred_file = join(pred_path, filename)
        image_file = join(image_path, filename)
        
        # 파일 존재 확인
        if not os.path.exists(gt_file):
            print(f"Warning: GT 파일 없음 - {gt_file}")
            continue
        
        if not os.path.exists(pred_file):
            print(f"Warning: 예측 파일 없음 - {pred_file}")
            continue
        
        # 이미지 및 마스크 로드
        gt_mask = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
        
        # 원본 이미지 로드 (있으면)
        if os.path.exists(image_file):
            image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        else:
            # 없으면 검은색 배경 사용
            image = np.zeros_like(gt_mask)
        
        # 크기 맞추기
        if pred_mask.shape != gt_mask.shape:
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
        
        if image.shape != gt_mask.shape:
            image = cv2.resize(image, (gt_mask.shape[1], gt_mask.shape[0]), 
                              interpolation=cv2.INTER_LINEAR)
        
        # 이진화 (threshold)
        _, gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
        _, pred_mask = cv2.threshold(pred_mask, 127, 255, cv2.THRESH_BINARY)
        
        # 오버레이 생성
        overlay_img, fp_mask, fn_mask, tp_mask = create_error_visualization(
            image, pred_mask, gt_mask, alpha=alpha
        )
        
        # 메트릭 계산
        metrics = compute_metrics(pred_mask, gt_mask)
        metrics['Filename'] = filename
        all_metrics.append(metrics)

        # Dice 점수를 파일명에 포함해 저장
        base, ext = os.path.splitext(filename)
        dice_score = metrics.get('Dice', 0.0)
        output_file = join(output_path, f"{base}_dice{dice_score:.4f}{ext}")
        cv2.imwrite(output_file, overlay_img)

        # 비교 이미지 생성 및 저장 (선택사항)
        if create_comparison:
            comparison_img = create_comparison_image(image, pred_mask / 255.0, gt_mask / 255.0)
            comparison_file = join(comparison_dir, f"{base}_dice{dice_score:.4f}{ext}")
            cv2.imwrite(comparison_file, comparison_img)
    
    # 통계 출력
    print("\n" + "="*80)
    print("통계 요약")
    print("="*80)
    
    total_tp = sum(m['TP'] for m in all_metrics)
    total_fp = sum(m['FP'] for m in all_metrics)
    total_fn = sum(m['FN'] for m in all_metrics)
    total_tn = sum(m['TN'] for m in all_metrics)
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) \
                 if (overall_precision + overall_recall) > 0 else 0
    
    print(f"전체 TP (True Positive):  {total_tp:,}")
    print(f"전체 FP (False Positive): {total_fp:,} (빨간색)")
    print(f"전체 FN (False Negative): {total_fn:,} (파란색)")
    print(f"전체 TN (True Negative):  {total_tn:,}")
    print(f"\nPrecision (정밀도): {overall_precision:.4f}")
    print(f"Recall (재현율):    {overall_recall:.4f}")
    print(f"F1-Score:           {overall_f1:.4f}")
    print("="*80)
    
    # 상위 FP, FN 많은 이미지들
    print("\n상위 FP가 많은 이미지 (모델이 과잉 예측):")
    top_fp = sorted(all_metrics, key=lambda x: x['FP'], reverse=True)[:5]
    for i, m in enumerate(top_fp, 1):
        print(f"{i}. {m['Filename']}: FP={m['FP']}, FN={m['FN']}, Precision={m['Precision']:.3f}")
    
    print("\n상위 FN이 많은 이미지 (모델이 예측 누락):")
    top_fn = sorted(all_metrics, key=lambda x: x['FN'], reverse=True)[:5]
    for i, m in enumerate(top_fn, 1):
        print(f"{i}. {m['Filename']}: FP={m['FP']}, FN={m['FN']}, Recall={m['Recall']:.3f}")
    
    print(f"\n✓ 오버레이 이미지가 저장되었습니다: {output_path}")
    if create_comparison:
        print(f"✓ 비교 이미지가 저장되었습니다: {comparison_dir}")


def main():
    parser = argparse.ArgumentParser(description='FP/FN 오버레이 시각화 도구')
    parser.add_argument('--gt-path', type=str, required=True, help='GT 마스크 디렉토리')
    parser.add_argument('--pred-path', type=str, required=True, help='예측 마스크 디렉토리')
    parser.add_argument('--image-path', type=str, help='원본 이미지 디렉토리 (선택사항)')
    parser.add_argument('--output-path', type=str, required=True, help='결과 저장 디렉토리')
    parser.add_argument('--alpha', type=float, default=0.5, help='오버레이 투명도 (0.0~1.0)')
    parser.add_argument('--comparison', action='store_true', help='비교 이미지 생성 여부')
    
    args = parser.parse_args()
    
    # image_path 기본값 설정
    if args.image_path is None:
        # pred_path의 상위 디렉토리에서 images 폴더 찾기
        pred_parent = os.path.dirname(args.pred_path)
        possible_image_path = join(pred_parent, '..', 'test_images')
        if os.path.exists(possible_image_path):
            args.image_path = possible_image_path
        else:
            args.image_path = None
            print("Warning: 원본 이미지 디렉토리를 찾을 수 없습니다. 검은색 배경으로 시각화합니다.")
    
    visualize_results(
        gt_path=args.gt_path,
        pred_path=args.pred_path,
        image_path=args.image_path,
        output_path=args.output_path,
        alpha=args.alpha,
        create_comparison=args.comparison
    )


if __name__ == '__main__':
    main()
