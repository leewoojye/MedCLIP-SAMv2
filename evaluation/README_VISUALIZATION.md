# FP/FN 오버레이 시각화 가이드

모델의 예측 결과와 Ground Truth의 차이를 시각적으로 확인하기 위한 두 가지 도구입니다.

## 📊 도구 1: `visualize_fp_fn.py` - 기본 오버레이 시각화

모든 테스트 이미지에 대해 FP/FN을 색깔별로 오버레이합니다.

### 색상 설명
- **빨간색 (FP - False Positive)**: 모델이 예측했지만 GT에는 없음 (과잉 예측)
- **파란색 (FN - False Negative)**: GT에는 있지만 모델이 예측하지 못함 (누락)
- **초록색 (TP - True Positive)**: 모델과 GT가 모두 일치 (정상 예측)

### 기본 사용법

```bash
python evaluation/visualize_fp_fn.py \
    --gt-path data/brain_tumors/test_masks \
    --pred-path sam_outputs/brain_tumors/masks \
    --image-path data/brain_tumors/test_images \
    --output-path evaluation/fp_fn_viz/brain_tumors \
    --alpha 0.5
```

### 옵션 설명

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--gt-path` | GT 마스크 디렉토리 | 필수 |
| `--pred-path` | 예측 마스크 디렉토리 | 필수 |
| `--image-path` | 원본 이미지 디렉토리 | 선택사항 |
| `--output-path` | 결과 저장 디렉토리 | 필수 |
| `--alpha` | 오버레이 투명도 (0.0-1.0) | 0.5 |
| `--comparison` | 비교 이미지 생성 여부 (원본/예측/GT를 나란히 표시) | 미포함 |

### 사용 예시

#### 예시 1: 뇌종양 - 기본 설정
```bash
python evaluation/visualize_fp_fn.py \
    --gt-path data/brain_tumors/test_masks \
    --pred-path sam_outputs/brain_tumors/masks \
    --image-path data/brain_tumors/test_images \
    --output-path evaluation/fp_fn_viz/brain_tumors \
    --alpha 0.5 \
    --comparison
```

#### 예시 2: 유방종양 - 강한 오버레이 (오류를 더 잘 보기 위해)
```bash
python evaluation/visualize_fp_fn.py \
    --gt-path data/breast_tumors/test_masks \
    --pred-path sam_outputs/breast_tumors/masks \
    --image-path data/breast_tumors/test_images \
    --output-path evaluation/fp_fn_viz/breast_tumors \
    --alpha 0.7
```

#### 예시 3: 폐 CT - 투명한 오버레이 (원본 이미지를 더 잘 보기 위해)
```bash
python evaluation/visualize_fp_fn.py \
    --gt-path data/lung_CT/test_masks \
    --pred-path sam_outputs/lung_CT/masks \
    --image-path data/lung_CT/test_images \
    --output-path evaluation/fp_fn_viz/lung_CT \
    --alpha 0.3
```

#### 예시 4: 서로 다른 단계 비교 (Coarse vs SAM)
```bash
# Coarse 출력 확인
python evaluation/visualize_fp_fn.py \
    --gt-path data/brain_tumors/test_masks \
    --pred-path coarse_outputs/brain_tumors/masks \
    --image-path data/brain_tumors/test_images \
    --output-path evaluation/fp_fn_viz/coarse_results

# SAM 출력 확인
python evaluation/visualize_fp_fn.py \
    --gt-path data/brain_tumors/test_masks \
    --pred-path sam_outputs/brain_tumors/masks \
    --image-path data/brain_tumors/test_images \
    --output-path evaluation/fp_fn_viz/sam_results
```

### 출력 결과

#### 1. 오버레이 이미지 (`output_path/`)
- 원본 이미지에 FP/FN을 색깔별로 오버레이한 이미지
- 해부학적 위치별 오류를 한눈에 파악 가능

#### 2. 비교 이미지 (`output_path/comparison/`) - `--comparison` 옵션 사용 시
- 원본 이미지 | 예측 마스크 | GT 마스크를 나란히 표시
- 각 단계별 결과를 명확하게 비교 가능

#### 3. 통계 정보 (콘솔 출력)
```
================================================================================
통계 요약
================================================================================
전체 TP (True Positive):  450000
전체 FP (False Positive): 12500 (빨간색)
전체 FN (False Negative): 8500 (파란색)
전체 TN (True Negative):  5000000

Precision (정밀도): 0.9731
Recall (재현율):    0.9812
F1-Score:           0.9771
================================================================================

상위 FP가 많은 이미지 (모델이 과잉 예측):
1. image_001.png: FP=450, FN=20, Precision=0.956
2. image_015.png: FP=380, FN=35, Precision=0.942
...

상위 FN이 많은 이미지 (모델이 예측 누락):
1. image_042.png: FP=50, FN=200, Recall=0.873
2. image_028.png: FP=65, FN=150, Recall=0.892
...
```

---

## 🎯 도구 2: `advanced_visualization.py` - 대시보드 시각화

여러 이미지를 한 페이지에 격자로 보여주는 대시보드를 생성합니다.

### 사용법

```bash
python evaluation/advanced_visualization.py \
    --gt-path data/brain_tumors/test_masks \
    --pred-path sam_outputs/brain_tumors/masks \
    --image-path data/brain_tumors/test_images \
    --output-path evaluation/dashboards \
    --num-images 6 \
    --dashboard
```

### 옵션

| 옵션 | 설명 |
|------|------|
| `--num-images` | 한 페이지에 표시할 이미지 개수 (기본값: 6) |
| `--dashboard` | 대시보드 생성 활성화 |

### 예시

```bash
# 9개 이미지 그리드로 대시보드 생성
python evaluation/advanced_visualization.py \
    --gt-path data/lung_Xray/test_masks \
    --pred-path sam_outputs/lung_Xray/masks \
    --image-path data/lung_Xray/test_images \
    --output-path evaluation/dashboards/lung_xray \
    --num-images 9 \
    --dashboard
```

---

## 🔄 자동화: 배치 스크립트

모든 데이터셋에 대해 자동으로 시각화를 수행하려면:

```bash
bash evaluation/visualize_examples.sh
```

이 스크립트는 다음을 순차적으로 실행합니다:
1. 뇌종양 시각화
2. 유방종양 시각화
3. Coarse vs SAM 비교
4. 폐 CT 고투명도 시각화

---

## 💡 해석 가이드

### FP (빨간색) 분석
- **많이 나타나는 위치**: 정상 조직과의 경계에서 모델이 혼동
- **해결 방법**: 
  - 경계 영역 학습 데이터 보강
  - Post-processing에서 작은 영역 제거
  - 모델 임계값 조정

### FN (파란색) 분석
- **많이 나타나는 위치**: 종양/병변의 중심 또는 가장자리
- **해결 방법**:
  - 해당 위치의 특징이 부족한 학습 데이터 확인
  - 모델의 감도(recall) 향상 필요
  - 데이터 augmentation 추가

### 해부학적 해석
- **폐 이미지**: FN이 주로 폐 가장자리에 나타나면 가우시안 필터 확인
- **종양**: FP가 necrosis 영역에 나타나면 정의 재확인 필요
- **일관된 패턴**: 특정 구조(혈관, 신경)에서 오류가 반복되면 학습 데이터 재검토

---

## 🚀 빠른 시작

```bash
# 1단계: 모든 데이터셋 시각화
bash evaluation/visualize_examples.sh

# 2단계: 결과 확인
# evaluation/fp_fn_viz/ 디렉토리에서 이미지 확인
# 웹 브라우저나 이미지 뷰어로 오픈

# 3단계: 상위 오류 이미지 분석
# 콘솔 출력에서 "상위 FP/FN" 이미지 확인
# 해당 이미지 시각적으로 검토
```

---

## 📝 참고사항

- **이미지 포맷**: PNG, JPG, JPEG 지원
- **마스크 형식**: 0-255 범위의 바이너리 마스크 (threshold=127)
- **크기 불일치**: 자동으로 GT 크기에 맞춤
- **메모리**: 대량의 고해상도 이미지는 배치 처리 권장

---

## 🔧 커스터마이징

### 색상 변경
`visualize_fp_fn.py`의 `create_error_visualization` 함수에서:
```python
# 색상 변경 (BGR 포맷)
overlay[fp_mask] = ... + np.array([0, 0, 255]) * alpha  # FP 색상
overlay[fn_mask] = ... + np.array([255, 0, 0]) * alpha  # FN 색상
overlay[tp_mask] = ... + np.array([0, 255, 0]) * alpha  # TP 색상
```

### 투명도 추천
- **0.3-0.4**: 원본 이미지를 주로 보고 싶을 때
- **0.5-0.6**: 균형잡힌 오버레이 (추천)
- **0.7-1.0**: 오류를 강조하고 싶을 때
