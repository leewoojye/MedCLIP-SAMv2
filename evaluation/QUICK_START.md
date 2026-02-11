# 🎯 FP/FN 시각화 - 빠른 시작 가이드

당신의 요청: **"모델 예측 부분과 GT 차이를 시각적으로 확인하고 싶은데 좋은 방법 줘봐"**

제가 만든 솔루션: **FP/FN 오버레이 시각화 도구**

---

## ⚡ 5분 안에 시작하기

### 1️⃣ 기본 사용법 (가장 간단한 방법)

```bash
cd /home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2

conda activate medclipsamv2

python evaluation/visualize_fp_fn.py \
    --gt-path data/brain_tumors/test_masks \
    --pred-path sam_outputs/data/brain_tumors/test_masks \
    --image-path data/brain_tumors/test_images \
    --output-path evaluation/results/brain_tumors \
    --alpha 0.5 \
    --comparison
```

**완료! 결과는 `evaluation/results/brain_tumors/` 에 저장됩니다.**

---

## 📊 출력 결과 해석

### 콘솔 출력 예시
```
================================================================================
통계 요약
================================================================================
전체 TP (True Positive):  580,591
전체 FP (False Positive): 540,654 (빨간색) ← 모델 과잉 예측
전체 FN (False Negative): 557,906 (파란색) ← 모델 누락

Precision (정밀도): 0.5178  (모델이 맞힐 확률)
Recall (재현율):    0.5100  (병변을 찾아낼 확률)
F1-Score:           0.5139
================================================================================

상위 FP가 많은 이미지 (모델이 과잉 예측):
1. 000071.png: FP=26027, FN=72, Precision=0.179
2. 000098.png: FP=20093, FN=2201, Precision=0.092
...

상위 FN이 많은 이미지 (모델이 예측 누락):
1. 000043.png: FP=822, FN=32289, Recall=0.122
2. 000041.png: FP=1839, FN=23511, Recall=0.514
...
```

### 생성된 이미지
- **오버레이 이미지**: 원본에 FP/FN을 색깔별로 표시
- **비교 이미지**: 원본 | 예측 | GT 를 나란히 표시

---

## 🎨 색상 가이드

| 색상 | 영어 | 의미 | 해석 |
|------|------|------|------|
| 🔴 빨강 | FP | False Positive | 모델만 예측 (과잉) |
| 🔵 파랑 | FN | False Negative | GT만 존재 (누락) |
| 🟢 초록 | TP | True Positive | 모두 일치 (정상) |

---

## 🔧 사용 상황별 설정

### 상황 1: 오류를 강하게 강조하고 싶을 때
```bash
--alpha 0.7
```
오버레이를 더 진하게 만들어서 오류가 눈에 띄도록

### 상황 2: 원본 의료 이미지를 주로 보고 싶을 때
```bash
--alpha 0.3
```
오버레이를 투명하게 만들어서 원본이 주로 보임

### 상황 3: 다양한 데이터셋 한번에 분석
```bash
bash evaluation/visualize_examples.sh
```
모든 데이터셋(뇌, 유방, 폐)을 자동으로 분석

### 상황 4: Coarse vs SAM 비교하고 싶을 때
```bash
# Coarse 단계
python evaluation/visualize_fp_fn.py \
    --gt-path data/brain_tumors/test_masks \
    --pred-path coarse_outputs/data/brain_tumors/masks \
    --output-path evaluation/results/coarse

# SAM 단계
python evaluation/visualize_fp_fn.py \
    --gt-path data/brain_tumors/test_masks \
    --pred-path sam_outputs/data/brain_tumors/test_masks \
    --output-path evaluation/results/sam
```

---

## 📈 모델 성능 읽는 방법

### 좋은 모델 (FP/FN이 적음)
```
✅ TP 많음 (초록색 많음)
✅ FP 적음 (빨간색 적음)
✅ FN 적음 (파란색 적음)
✅ Precision/Recall 모두 0.9 이상
```

### 개선 필요한 모델
```
❌ FP 많음 (빨간색 많음) → 모델이 너무 민감 (과잉 예측)
❌ FN 많음 (파란색 많음) → 모델이 병변을 놓침
❌ Precision 낮음 → 오진이 많음
❌ Recall 낮음 → 병변 검출률이 낮음
```

---

## 🔍 해부학적 오류 분석

### FP(빨간색)가 많은 위치
- **경계 영역**: 영상 후처리(morphological operation) 추가
- **노이즈 영역**: 데이터 정제 또는 threshold 조정
- **정상 구조(혈관 등)**: 학습 데이터에서 구분 표시 필요

### FN(파란색)이 많은 위치
- **종양 중심**: 모델이 주요 특징을 놓침 → 학습 데이터 점검
- **가장자리**: Boundary detection 성능 부족 → 특화 학습 필요
- **작은 병변**: 해상도 부족 → 고해상도 입력 또는 다중 스케일 처리

---

## 📁 파일 구조

생성되는 디렉토리 구조:
```
evaluation/
├── results/
│   ├── brain_tumors/
│   │   ├── 000001.png (오버레이)
│   │   ├── 000002.png
│   │   ├── comparison/
│   │   │   ├── 000001.png (비교)
│   │   │   └── 000002.png
│   │   └── ...
│   ├── breast_tumors/
│   └── lung_CT/
├── dashboards/
│   ├── dashboard_batch_000.png
│   ├── dashboard_batch_001.png
│   └── ...
└── visualize_fp_fn.py (메인 스크립트)
```

---

## 💡 추가 팁

### 1. 상위 오류 이미지 먼저 분석
콘솔 출력의 "상위 FP/FN" 이미지를 먼저 확인하면 주요 문제 파악이 빠름

### 2. 투명도(alpha) 값 추천
- **의료 영상 분석**: 0.5 (균형잡힌 비교)
- **오류 강조**: 0.7-0.8 (오류를 크게 보임)
- **원본 중시**: 0.3-0.4 (원본이 주로 보임)

### 3. 배치 처리 (많은 이미지)
```bash
# 전체 데이터셋 한번에 분석
for dataset in brain_tumors breast_tumors lung_CT lung_Xray; do
    python evaluation/visualize_fp_fn.py \
        --gt-path data/$dataset/test_masks \
        --pred-path sam_outputs/data/$dataset/test_masks \
        --image-path data/$dataset/test_images \
        --output-path evaluation/results/$dataset \
        --alpha 0.5 \
        --comparison
done
```

---

## 🚀 실행 가능한 전체 분석 스크립트

```bash
# 한 줄로 모든 것을 분석
bash evaluation/run_full_analysis.sh
```

이 스크립트는:
- ✅ 모든 데이터셋 시각화
- ✅ 처리 단계별(Coarse vs SAM) 비교
- ✅ 통계 생성
- ✅ 대시보드 생성

---

## ❓ 자주 묻는 질문

**Q: 스크립트를 실행해도 아무것도 표시되지 않습니다**
A: 다음을 확인하세요:
1. 파일 경로가 정확한지 (`--gt-path`, `--pred-path` 등)
2. 이미지 파일이 실제로 존재하는지
3. conda 환경이 활성화되었는지 (`conda activate medclipsamv2`)

**Q: 이미지가 흑백으로만 보입니다**
A: 마스크 파일이 이진(binary)이어야 합니다. 비교 이미지(`--comparison`)를 생성하면 컬러로 볼 수 있습니다.

**Q: 메모리 부족 에러가 발생합니다**
A: 배치 크기를 줄이거나 다음과 같이 부분적으로 처리하세요:
```bash
ls data/brain_tumors/test_masks/ | head -100 | while read f; do
    # 처리...
done
```

**Q: 다른 데이터셋에도 적용할 수 있나요?**
A: 네! `--gt-path`, `--pred-path`, `--image-path`, `--output-path`만 변경하면 됩니다.

---

## 📞 다음 단계

1. **시각화 결과 분석**: 오버레이 이미지에서 주요 오류 위치 파악
2. **통계 검토**: Precision, Recall, F1-Score 확인
3. **개선 전략 수립**: FP/FN 위치별 해결 방안 계획
4. **모델 미세 조정**: 필요하면 학습 데이터 보강 또는 threshold 조정

---

**축하합니다! 이제 모델의 문제점을 시각적으로 명확하게 볼 수 있습니다! 🎉**
