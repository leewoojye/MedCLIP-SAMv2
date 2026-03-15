"""
히스토그램 평활화 실험
뇌종양 테스트 이미지에 히스토그램 평활화를 적용하고 비교
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def apply_histogram_equalization(image):
    """그레이스케일 이미지에 히스토그램 평활화 적용"""
    return cv2.equalizeHist(image)


def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    CLAHE(Contrast Limited Adaptive Histogram Equalization) 적용
    - clip_limit: 대비 제한 값 (낮을수록 더 보수적)
    - tile_grid_size: 타일 그리드 크기
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def compute_histogram(image):
    """이미지의 히스토그램 계산"""
    return cv2.calcHist([image], [0], None, [256], [0, 256])


def main():
    # 이미지 경로 설정
    img_dir = Path("ld_paper/normal_sample")
    img_files = list(img_dir.glob("*.png"))
    
    if not img_files:
        print(f"에러: {img_dir}에 이미지 파일이 없습니다.")
        return
    
    # 첫 번째 이미지 선택 (또는 원하는 이미지로 변경)
    img_path = img_files[3]
    print(f"처리 이미지: {img_path}")
    
    # 이미지 로드
    original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    
    if original is None:
        print(f"에러: 이미지를 로드할 수 없습니다: {img_path}")
        return
    
    print(f"이미지 크기: {original.shape}")
    print(f"이미지 데이터 타입: {original.dtype}")
    print(f"픽셀값 범위: [{original.min()}, {original.max()}]")
    
    # 히스토그램 평활화 적용
    equalized = apply_histogram_equalization(original)
    
    # CLAHE 적용
    clahe_img = apply_clahe(original, clip_limit=2.0, tile_grid_size=(8, 8))
    
    # 히스토그램 계산
    hist_original = compute_histogram(original)
    hist_equalized = compute_histogram(equalized)
    hist_clahe = compute_histogram(clahe_img)
    
    # 시각화
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f"히스토그램 평활화 실험 - {img_path.name}", fontsize=16)
    
    # 원본 이미지
    axes[0, 0].imshow(original, cmap='gray')
    axes[0, 0].set_title("원본 이미지")
    axes[0, 0].axis('off')
    
    # 평활화된 이미지
    axes[0, 1].imshow(equalized, cmap='gray')
    axes[0, 1].set_title("히스토그램 평활화")
    axes[0, 1].axis('off')
    
    # CLAHE 이미지
    axes[0, 2].imshow(clahe_img, cmap='gray')
    axes[0, 2].set_title("CLAHE")
    axes[0, 2].axis('off')
    
    # 원본 히스토그램
    axes[1, 0].plot(hist_original, color='black', linewidth=0.7)
    axes[1, 0].set_title("원본 히스토그램")
    axes[1, 0].set_xlim([0, 256])
    axes[1, 0].grid(True, alpha=0.3)
    
    # 평활화 히스토그램
    axes[1, 1].plot(hist_equalized, color='blue', linewidth=0.7)
    axes[1, 1].set_title("평활화 히스토그램")
    axes[1, 1].set_xlim([0, 256])
    axes[1, 1].grid(True, alpha=0.3)
    
    # CLAHE 히스토그램
    axes[1, 2].plot(hist_clahe, color='green', linewidth=0.7)
    axes[1, 2].set_title("CLAHE 히스토그램")
    axes[1, 2].set_xlim([0, 256])
    axes[1, 2].grid(True, alpha=0.3)
    
    # 비교: 3가지 히스토그램 겹쳐서 표시
    axes[2, 0].plot(hist_original, color='black', label='원본', linewidth=1, alpha=0.7)
    axes[2, 0].plot(hist_equalized, color='blue', label='평활화', linewidth=1, alpha=0.7)
    axes[2, 0].plot(hist_clahe, color='green', label='CLAHE', linewidth=1, alpha=0.7)
    axes[2, 0].set_title("히스토그램 비교")
    axes[2, 0].set_xlim([0, 256])
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 차이 이미지 (평활화 - 원본)
    diff_eq = cv2.absdiff(equalized, original)
    axes[2, 1].imshow(diff_eq, cmap='hot')
    axes[2, 1].set_title("차이: 평활화 - 원본")
    axes[2, 1].axis('off')
    plt.colorbar(axes[2, 1].imshow(diff_eq, cmap='hot'), ax=axes[2, 1])
    
    # 차이 이미지 (CLAHE - 원본)
    diff_clahe = cv2.absdiff(clahe_img, original)
    axes[2, 2].imshow(diff_clahe, cmap='hot')
    axes[2, 2].set_title("차이: CLAHE - 원본")
    axes[2, 2].axis('off')
    plt.colorbar(axes[2, 2].imshow(diff_clahe, cmap='hot'), ax=axes[2, 2])
    
    plt.tight_layout()
    
    # 결과 저장
    output_path = "histogram_equalization_result.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n결과 저장: {output_path}")
    
    # 통계 정보 출력
    print("\n=== 통계 정보 ===")
    print(f"원본 - 평균: {original.mean():.2f}, 표준편차: {original.std():.2f}")
    print(f"평활화 - 평균: {equalized.mean():.2f}, 표준편차: {equalized.std():.2f}")
    print(f"CLAHE - 평균: {clahe_img.mean():.2f}, 표준편차: {clahe_img.std():.2f}")
    
    # 개별 이미지 저장
    cv2.imwrite("original_image.png", original)
    cv2.imwrite("equalized_image.png", equalized)
    cv2.imwrite("clahe_image.png", clahe_img)
    print("\n개별 이미지 저장 완료:")
    print("- original_image.png")
    print("- equalized_image.png")
    print("- clahe_image.png")
    
    plt.show()


if __name__ == "__main__":
    main()
