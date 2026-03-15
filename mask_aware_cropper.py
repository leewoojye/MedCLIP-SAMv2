"""
Mask-aware cropping strategy
논문의 의도: ROI 주변 배경 제거 + 정보 손실 최소화
"""
import numpy as np
from PIL import Image


class MaskAwareCropper:
    """마스크 기반 스마트 크롭핑"""
    
    def __init__(self, target_size=224, padding_ratio=0.15):
        """
        Args:
            target_size: 목표 최소 크기
            padding_ratio: ROI 주변 패딩 비율 (0.15 = 15% 추가 margin)
        """
        self.target_size = target_size
        self.padding_ratio = padding_ratio
    
    def get_roi_bbox(self, mask):
        """
        마스크에서 관심 영역(ROI) 바운딩 박스 찾기
        
        Returns:
            (y_min, y_max, x_min, x_max)
        """
        # 마스크 이진화
        binary_mask = mask > 0.5
        
        if not binary_mask.any():
            # 마스크가 비어있으면 전체 이미지 반환
            h, w = mask.shape
            return 0, h, 0, w
        
        # ROI 찾기
        coords = np.where(binary_mask)
        y_min, y_max = coords[0].min(), coords[0].max()
        x_min, x_max = coords[1].min(), coords[1].max()
        
        # ROI 높이/너비
        roi_h = y_max - y_min
        roi_w = x_max - x_min
        
        # 패딩 추가 (배경 컨텍스트 포함)
        pad_h = int(roi_h * self.padding_ratio)
        pad_w = int(roi_w * self.padding_ratio)
        
        y_min = max(0, y_min - pad_h)
        y_max = min(mask.shape[0], y_max + pad_h)
        x_min = max(0, x_min - pad_w)
        x_max = min(mask.shape[1], x_max + pad_w)
        
        return y_min, y_max, x_min, x_max
    
    def crop_with_mask(self, image, mask, strategy="roi_based"):
        """
        마스크를 활용한 스마트 크롭
        
        Args:
            strategy:
                - "roi_based": ROI 주변에서 crop (권장)
                - "fixed_224": 224x224로 고정 (마지막 수단)
                - "preserve": 원본 크기 유지 (정보 손실 0)
        """
        if strategy == "roi_based":
            return self._roi_based_crop(image, mask)
        elif strategy == "fixed_224":
            return self._fixed_crop(image, mask)
        elif strategy == "preserve":
            return self._preserve_size(image, mask)
        else:
            return self._roi_based_crop(image, mask)
    
    def _roi_based_crop(self, image, mask):
        """
        ROI 기반 크롭 (권장)
        - ROI + 패딩을 찾아서 그 영역 crop
        - 배경 제거 + 정보 손실 최소화
        """
        y_min, y_max, x_min, x_max = self.get_roi_bbox(mask)
        
        cropped_img = image[y_min:y_max, x_min:x_max]
        cropped_mask = mask[y_min:y_max, x_min:x_max]
        
        # 크롭된 이미지가 target_size보다 작으면 패딩
        h, w = cropped_img.shape
        if h < self.target_size or w < self.target_size:
            cropped_img = self._pad_to_size(cropped_img, self.target_size)
            cropped_mask = self._pad_to_size(cropped_mask, self.target_size)
        
        return cropped_img, cropped_mask
    
    def _fixed_crop(self, image, mask):
        """224x224 고정 크롭 (논문 방식)"""
        h, w = image.shape
        
        if h < self.target_size or w < self.target_size:
            image = self._pad_to_size(image, self.target_size)
            mask = self._pad_to_size(mask, self.target_size)
            return image, mask
        
        # ROI 중심에서 224x224 추출 (배경 제거 의도 구현)
        y_min, y_max, x_min, x_max = self.get_roi_bbox(mask)
        roi_cy = (y_min + y_max) // 2
        roi_cx = (x_min + x_max) // 2
        
        # ROI 중심에서 224x224로 crop
        start_y = max(0, min(roi_cy - self.target_size // 2, h - self.target_size))
        start_x = max(0, min(roi_cx - self.target_size // 2, w - self.target_size))
        
        return (image[start_y:start_y + self.target_size, start_x:start_x + self.target_size],
                mask[start_y:start_y + self.target_size, start_x:start_x + self.target_size])
    
    def _preserve_size(self, image, mask):
        """원본 크기 유지 + 패딩으로 통일"""
        h, w = image.shape
        max_size = max(h, w)
        
        # 최소 target_size 이상이 되도록
        target = max(self.target_size, max_size)
        
        image = self._pad_to_size(image, target)
        mask = self._pad_to_size(mask, target)
        
        return image, mask
    
    def _pad_to_size(self, img, target_size):
        """이미지를 패딩으로 target_size로 확장"""
        h, w = img.shape
        
        if h >= target_size and w >= target_size:
            return img
        
        pad_h = max(0, target_size - h)
        pad_w = max(0, target_size - w)
        
        # 중앙에 배치
        pad_h_before = pad_h // 2
        pad_h_after = pad_h - pad_h_before
        pad_w_before = pad_w // 2
        pad_w_after = pad_w - pad_w_before
        
        padded = np.pad(
            img,
            ((pad_h_before, pad_h_after), (pad_w_before, pad_w_after)),
            mode='constant',
            constant_values=0
        )
        
        return padded[:target_size, :target_size]


# 권장 사용법
"""
전략별 결정:

1️⃣ ROI-based (권장) ⭐⭐⭐
   - 배경 제거 (논문 의도)
   - 정보 손실 최소
   - 크기 가변적
   
2️⃣ Fixed-224 (데이터 증강)
   - 논문과 동일
   - ROI 중심 기반이라 합리적
   
3️⃣ Preserve (정보 100% 보존)
   - 정보 손실 0
   - 메모리 사용 증가
   - 모델이 다양한 크기 처리 필요

당신의 경우: ROI-based 추천!
- 마스크가 있으니 ROI 정확하게 파악 가능
- 배경 제거 + 정보 손실 최소화 달성
- 크기 가변성 자동 처리
"""
