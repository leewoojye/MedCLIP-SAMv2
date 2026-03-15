"""
스마트한 이미지 리사이징/크로핑 전략들
데이터 손실을 최소화하면서 224x224로 통일
"""
import numpy as np
from PIL import Image

class SmartImagePreprocessor:
    """여러 리사이징 전략 제공"""
    
    def __init__(self, target_size=224):
        self.target_size = target_size
    
    # === 전략 1: Center Crop (권장 - 간단하고 안정적) ===
    def center_crop(self, image):
        """
        중앙의 224x224만 추출
        장점: 가장 간단, 중요 부분이 보통 중앙
        단점: 주변 정보 손실
        """
        w, h = image.size
        left = (w - self.target_size) // 2
        top = (h - self.target_size) // 2
        right = left + self.target_size
        bottom = top + self.target_size
        return image.crop((left, top, right, bottom))
    
    # === 전략 2: Random Crop (권장 - 최고의 데이터 활용) ===
    def random_crop(self, image, seed=None):
        """
        임의의 224x224 영역을 추출 (학습 중 데이터 증강 효과)
        장점: 매 epoch마다 다른 부분 학습 → 데이터 손실 최소화
        단점: 학습 불안정성 가능 (seed 고정하면 해결)
        """
        w, h = image.size
        
        if seed is not None:
            np.random.seed(seed)
        
        # 224x224가 들어갈 수 있는 영역 내에서 임의로 선택
        left = np.random.randint(0, w - self.target_size + 1)
        top = np.random.randint(0, h - self.target_size + 1)
        right = left + self.target_size
        bottom = top + self.target_size
        
        return image.crop((left, top, right, bottom))
    
    # === 전략 3: Resize with Aspect Preservation (데이터 손실 아주 적음) ===
    def resize_with_aspect_ratio(self, image):
        """
        원본 비율 유지하면서 리사이즈 후 필요시 패딩
        모든 정보 보존, 가장 손실 없음
        """
        w, h = image.size
        
        # 단축변을 224로 설정 (정사각형이므로 그냥 리사이즈)
        scale = self.target_size / min(w, h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # LANCZOS: 고품질 다운샘플링
        resized = image.resize((new_w, new_h), Image.LANCZOS)
        
        # 정사각형 이미지이므로 센터 크롭으로 224x224로
        if new_w > self.target_size or new_h > self.target_size:
            return self.center_crop(resized)
        
        return resized
    
    # === 전략 4: Multi-Scale Patches (최고의 정확도) ===
    def extract_multiple_patches(self, image, num_patches=4):
        """
        하나의 이미지에서 여러 224x224 패치 추출
        모델이 이미지의 다양한 부분 학습
        """
        w, h = image.size
        patches = []
        
        if num_patches == 4:
            # 4개의 코너 패치 + 센터 = 5개
            offsets = [
                (0, 0),  # top-left
                (w - self.target_size, 0),  # top-right
                (0, h - self.target_size),  # bottom-left
                (w - self.target_size, h - self.target_size),  # bottom-right
                ((w - self.target_size) // 2, (h - self.target_size) // 2),  # center
            ]
        else:
            # 그리드 방식
            step = (w - self.target_size) // (num_patches - 1)
            offsets = [(i * step, 0) for i in range(num_patches)]
        
        for left, top in offsets:
            patch = image.crop((left, top, left + self.target_size, top + self.target_size))
            patches.append(patch)
        
        return patches
    
    # === 전략 5: Gaussian Weighted Crop (최적 타협) ===
    def gaussian_weighted_crop(self, image, sigma=None):
        """
        이미지 중앙에 가우시안 가중치 적용 후 크롭
        중요한 중앙 부분 선호하지만 가장자리 정보도 포함 가능
        """
        if sigma is None:
            sigma = min(image.size) / 6  # 이미지 크기에 따라 자동 조정
        
        w, h = image.size
        max_left = w - self.target_size
        max_top = h - self.target_size
        
        # 가우시안 분포로부터 위치 샘플링
        # 중앙(0.5)을 중심으로, sigma만큼 퍼짐
        left = np.clip(
            np.random.normal(max_left / 2, sigma),
            0, max_left
        ).astype(int)
        top = np.clip(
            np.random.normal(max_top / 2, sigma),
            0, max_top
        ).astype(int)
        
        return image.crop((left, top, left + self.target_size, top + self.target_size))


# === 추천 사용법 ===
def recommended_strategy():
    """
    현재 데이터셋에 최적의 전략:
    
    1. 검증 셋: center_crop() ← 일관성 필요
    2. 학습 셋: random_crop() ← 데이터 증강 + 최대 정보 활용
    3. 못 쓸 이미지 걱정 없음: 모든 이미지가 224보다 큼
    
    구현에서:
    - Train DataLoader: random_crop 사용 (매 epoch마다 다른 crop)
    - Val/Test DataLoader: center_crop 사용 (일관된 평가)
    """
    print("""
    === 권장 전략 ===
    
    1️⃣  간단함 우선 → center_crop()
       - 구현 최소, 데이터 손실 적음
       - 정사각형 이미지에 최적
    
    2️⃣  최고 성능 → random_crop() (권장)
       - 매 epoch마다 다른 부분 학습
       - 데이터 증강 효과 좋음
       - 데이터 손실 최소화
    
    3️⃣  최고 정밀도 → extract_multiple_patches()
       - 학습 용량 증가 (1개 이미지 → 5개 패치)
       - 모델이 다양한 부분 전문화
       - 메모리 5배 필요
    
    현재 데이터: 256~512px 정사각형 (모두 224보다 큼)
    → 패딩/못 쓸 이미지 0개, 자유롭게 선택 가능!
    """)

if __name__ == "__main__":
    recommended_strategy()
