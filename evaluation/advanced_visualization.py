"""
고급 오류 분석 시각화
- 히트맵으로 FP/FN 위치 표시
- 영역별 오류 통계
- 여러 이미지 오버레이를 한 페이지에 표시
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import argparse
from tqdm import tqdm
from matplotlib.gridspec import GridSpec

join = os.path.join


def create_error_heatmap(pred_mask, gt_mask, window_size=32):
    """
    윈도우 기반 오류 히트맵 생성
    각 윈도우에서 FP/FN의 비율을 시각화
    """
    pred_mask = (pred_mask > 127).astype(np.float32)
    gt_mask = (gt_mask > 127).astype(np.float32)
    
    h, w = pred_mask.shape
    heatmap = np.zeros((h, w))
    
    for y in range(0, h - window_size, window_size):
        for x in range(0, w - window_size, window_size):
            window_pred = pred_mask[y:y+window_size, x:x+window_size]
            window_gt = gt_mask[y:y+window_size, x:x+window_size]
            
            # 이 윈도우에서의 오류 비율
            fp = np.sum((window_pred == 1) & (window_gt == 0))
            fn = np.sum((window_pred == 0) & (window_gt == 1))
            tp = np.sum((window_pred == 1) & (window_gt == 1))
            
            total_positives = tp + fp + fn
            if total_positives > 0:
                error_ratio = (fp + fn) / total_positives
                heatmap[y:y+window_size, x:x+window_size] = error_ratio
    
    return heatmap


def create_regional_analysis(pred_mask, gt_mask, image):
    """
    영역별 분석 정보 반환
    """
    pred_mask = (pred_mask > 127).astype(np.int32)
    gt_mask = (gt_mask > 127).astype(np.int32)
    
    # 경계 영역 감지 (GT 경계 주변 10px)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
    gt_dilated = cv2.dilate(gt_mask * 255, kernel, iterations=1).astype(np.int32) / 255
    gt_eroded = cv2.erode(gt_mask * 255, kernel, iterations=1).astype(np.int32) / 255
    boundary_mask = (gt_dilated - gt_eroded) > 0
    
    # 영역별 오류
    interior_mask = (gt_mask == 1) & ~boundary_mask
    
    fp_interior = np.sum((pred_mask == 1) & (gt_mask == 0) & interior_mask)
    fp_boundary = np.sum((pred_mask == 1) & (gt_mask == 0) & boundary_mask)
    fp_background = np.sum((pred_mask == 1) & (gt_mask == 0) & ~boundary_mask & ~interior_mask)
    
    fn_interior = np.sum((pred_mask == 0) & (gt_mask == 1) & interior_mask)
    fn_boundary = np.sum((pred_mask == 0) & (gt_mask == 1) & boundary_mask)
    
    return {
        'fp_interior': fp_interior,
        'fp_boundary': fp_boundary,
        'fp_background': fp_background,
        'fn_interior': fn_interior,
        'fn_boundary': fn_boundary,
        'boundary_mask': boundary_mask,
        'interior_mask': interior_mask
    }


def create_multi_image_dashboard(gt_path, pred_path, image_path, output_path, num_images=6):
    """
    여러 이미지를 한 페이지에 시각화하는 대시보드 생성
    """
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    mask_files = os.listdir(pred_path)
    mask_files = [f for f in mask_files if f.endswith(('.png', '.jpg', '.jpeg'))]
    mask_files.sort()
    
    # 배치로 처리
    num_batches = (len(mask_files) + num_images - 1) // num_images
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * num_images
        end_idx = min(start_idx + num_images, len(mask_files))
        batch_files = mask_files[start_idx:end_idx]
        
        # 대시보드 생성
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        for idx, filename in enumerate(batch_files):
            gt_file = join(gt_path, filename)
            pred_file = join(pred_path, filename)
            image_file = join(image_path, filename) if image_path else None
            
            if not os.path.exists(gt_file) or not os.path.exists(pred_file):
                continue
            
            gt_mask = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
            pred_mask = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE)
            
            if image_file and os.path.exists(image_file):
                image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            else:
                image = np.zeros_like(gt_mask)
            
            if pred_mask.shape != gt_mask.shape:
                pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]))
            
            if image.shape != gt_mask.shape:
                image = cv2.resize(image, (gt_mask.shape[1], gt_mask.shape[0]))
            
            _, gt_mask = cv2.threshold(gt_mask, 127, 255, cv2.THRESH_BINARY)
            _, pred_mask = cv2.threshold(pred_mask, 127, 255, cv2.THRESH_BINARY)
            
            # FP/FN 계산
            fp_mask = (pred_mask > 127) & (gt_mask < 127)
            fn_mask = (pred_mask < 127) & (gt_mask > 127)
            tp_mask = (pred_mask > 127) & (gt_mask > 127)
            
            # 컬러 이미지로 변환
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).astype(np.float32)
            result[fp_mask] = [0, 0, 255]  # 빨강 (FP)
            result[fn_mask] = [255, 0, 0]  # 파랑 (FN)
            result[tp_mask] = [0, 255, 0]  # 초록 (TP)
            result = np.uint8(result)
            
            # Subplot에 표시
            ax = fig.add_subplot(gs[idx // 3, idx % 3])
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            ax.imshow(result_rgb)
            
            # 통계
            fp_count = np.sum(fp_mask)
            fn_count = np.sum(fn_mask)
            tp_count = np.sum(tp_mask)
            
            precision = tp_count / (tp_count + fp_count) if (tp_count + fp_count) > 0 else 0
            recall = tp_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0
            
            title = f"{filename}\nFP:{fp_count} FN:{fn_count} TP:{tp_count}\nPrec:{precision:.3f} Rec:{recall:.3f}"
            ax.set_title(title, fontsize=9)
            ax.axis('off')
        
        # 범례 추가
        fig.text(0.5, 0.02, '빨강(FP): 모델만 예측 | 파랑(FN): GT만 존재 | 초록(TP): 일치', 
                ha='center', fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 저장
        output_file = join(output_path, f'dashboard_batch_{batch_idx:03d}.png')
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        print(f"Dashboard saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='고급 오류 분석 시각화')
    parser.add_argument('--gt-path', type=str, required=True, help='GT 마스크 디렉토리')
    parser.add_argument('--pred-path', type=str, required=True, help='예측 마스크 디렉토리')
    parser.add_argument('--image-path', type=str, help='원본 이미지 디렉토리')
    parser.add_argument('--output-path', type=str, required=True, help='결과 저장 디렉토리')
    parser.add_argument('--num-images', type=int, default=6, help='한 페이지에 표시할 이미지 개수')
    parser.add_argument('--dashboard', action='store_true', help='대시보드 생성 여부')
    
    args = parser.parse_args()
    
    if args.dashboard:
        create_multi_image_dashboard(
            gt_path=args.gt_path,
            pred_path=args.pred_path,
            image_path=args.image_path,
            output_path=args.output_path,
            num_images=args.num_images
        )


if __name__ == '__main__':
    main()
