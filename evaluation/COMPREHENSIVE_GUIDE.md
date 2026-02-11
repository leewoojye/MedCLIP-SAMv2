# 🎯 MedCLIP-SAMv2 모델 예측 분석 도구 종합 가이드

## 📌 요청 사항
> "모델 예측 부분과 GT 차이를 시각적으로 확인하고 싶은데, FP(빨간색), FN(파란색) 영역을 색깔별로 오버레이 해서 모델이 구체적으로 어떤 해부학적 위치에서 헷갈려 하는지 한눈에 파악하고 싶습니다."

## ✅ 제공된 솔루션

### 1. **FP/FN 오버레이 시각화 도구** (`visualize_fp_fn.py`)
모든 테스트 이미지에 대해 자동으로:
- 예측 마스크와 GT 마스크를 비교
- FP(False Positive): **빨간색** - 모델이 예측했지만 GT에는 없음
- FN(False Negative): **파란색** - GT에는 있지만 모델이 예측하지 못함
- TP(True Positive): **초록색** - 모두 일치한 부분
- 원본 이미지에 오버레이
- 통계 정보(Precision, Recall, F1-Score) 자동 계산
- 상위 FP/FN 이미지 자동 식별

### 2. **고급 시각화 도구** (`advanced_visualization.py`)
- 여러 이미지를 한 페이지 격자에 표시 (대시보드)
- 윈도우 기반 오류 히트맵
- 영역별 오류 분석 (경계 vs 내부)

### 3. **배치 처리 스크립트** (`visualize_examples.sh`, `run_full_analysis.sh`)
- 모든 데이터셋 자동 분석
- 처리 단계별 비교 (Coarse vs SAM)
- 일괄 처리로 시간 절약

### 4. **비교 이미지** (`--comparison` 옵션)
- 원본 이미지 | 예측 마스크 | GT 마스크를 나란히 표시
- 각 단계별 차이를 명확하게 비교 가능

---

## 🚀 빠른 시작 (3가지 방법)

### 방법 1: 한 줄 명령어 (가장 빠름)
```bash
cd /home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2
conda activate medclipsamv2

python evaluation/visualize_fp_fn.py \
    --gt-path data/brain_tumors/test_masks \
    --pred-path sam_outputs/data/brain_tumors/test_masks \
    --image-path data/brain_tumors/test_images \
    --output-path evaluation/results/brain_tumors \
    --alpha 0.5 --comparison
```

### 방법 2: 배치 스크립트 (모든 데이터셋)
```bash
bash evaluation/visualize_examples.sh
```

### 방법 3: 전체 분석 (권장)
```bash
bash evaluation/run_full_analysis.sh
```
자동으로:
- 모든 데이터셋 시각화
- 처리 단계 비교
- 통계 생성
- 대시보드 생성

---

## 📊 생성되는 결과물

### 1. 오버레이 이미지
```
evaluation/results/brain_tumors/
├── 000001.png  (원본 + FP/FN 오버레이)
├── 000002.png
├── 000003.png
└── ...
```
- 원본 의료 이미지 위에 오류를 색깔별로 표시
- 해부학적 위치별 오류 패턴을 한눈에 파악 가능

### 2. 비교 이미지
```
evaluation/results/brain_tumors/comparison/
├── 000001.png  (원본 | 예측 | GT)
├── 000002.png
└── ...
```
- 각 단계별 결과를 나란히 비교
- 정량적 지표 표시 (Precision, Recall 등)

### 3. 통계 정보 (콘솔 출력)
```
================================================================================
통계 요약
================================================================================
전체 TP (True Positive):  580,591       ✅ 맞게 예측
전체 FP (False Positive): 540,654       ❌ 과잉 예측 (빨간색)
전체 FN (False Negative): 557,906       ❌ 누락 (파란색)
전체 TN (True Negative):  26,678,381    ✅ 맞게 제외

Precision (정밀도): 0.5178  (모델이 맞힐 확률)
Recall (재현율):    0.5100  (병변을 찾아낼 확률)
F1-Score:           0.5139  (정밀도와 재현율의 조화평균)

상위 FP가 많은 이미지 (모델이 과잉 예측):
1. 000071.png: FP=26027, FN=72, Precision=0.179
2. 000098.png: FP=20093, FN=2201, Precision=0.092

상위 FN이 많은 이미지 (모델이 예측 누락):
1. 000043.png: FP=822, FN=32289, Recall=0.122
2. 000041.png: FP=1839, FN=23511, Recall=0.514
```

### 4. 대시보드
```
evaluation/results/dashboards/
├── dashboard_batch_000.png  (6-9개 이미지 격자)
├── dashboard_batch_001.png
└── ...
```

---

## 🎨 색상 해석

| 색상 | 픽셀 상태 | 의미 | 원인 분석 |
|------|---------|------|---------|
| 🔴 **빨강** | FP | 모델만 예측 | 모델이 너무 민감하거나 정상 조직을 병변으로 착각 |
| 🔵 **파랑** | FN | GT만 존재 | 모델이 병변을 놓치거나 특징을 제대로 학습하지 못함 |
| 🟢 **초록** | TP | 모두 일치 | 올바른 예측 |
| **검은색** | TN | 모두 음수 | 정상 영역을 제대로 제외 |

---

## 🔍 해부학적 위치별 오류 분석 방법

### 단계 1: 오버레이 이미지 확인
```bash
# 이미지 뷰어로 확인
eog evaluation/results/brain_tumors/*.png
```

### 단계 2: 상위 오류 이미지 분석
콘솔에 출력된 상위 FP/FN 이미지부터 집중 분석

### 단계 3: 패턴 식별
- **경계 영역에 FP**: 모델이 경계 근처를 과하게 포함
- **중심부에 FN**: 모델이 병변 중심을 놓침
- **특정 위치에 집중**: 해부학적 특정 구조에서 오류

### 단계 4: 개선 방안 수립
- **FP 많음**: Threshold 조정, Post-processing 강화
- **FN 많음**: 학습 데이터 보강, 모델 미세 조정

---

## 📋 전체 옵션 설명

### `visualize_fp_fn.py` 옵션

```bash
python evaluation/visualize_fp_fn.py \
    --gt-path DATA_PATH              # [필수] GT 마스크 디렉토리
    --pred-path PRED_PATH            # [필수] 예측 마스크 디렉토리
    --image-path IMAGE_PATH          # [선택] 원본 이미지 디렉토리
    --output-path OUTPUT_PATH        # [필수] 결과 저장 디렉토리
    --alpha 0.5                      # 투명도 (0.0~1.0, 기본: 0.5)
    --comparison                     # 비교 이미지 생성 여부 (플래그)
```

### 투명도(alpha) 추천값

| 값 | 용도 | 설명 |
|----|------|------|
| **0.3** | 원본 중시 | 원본 의료 이미지를 주로 관찰 |
| **0.5** | 균형 | 원본과 오류를 모두 적절히 관찰 (🌟 추천) |
| **0.7** | 오류 강조 | 오류 위치를 명확하게 강조 |
| **1.0** | 오류만 | 오류만 순수하게 표시 |

---

## 💡 상황별 사용 예시

### 📌 상황 1: 뇌종양 기본 분석
```bash
python evaluation/visualize_fp_fn.py \
    --gt-path data/brain_tumors/test_masks \
    --pred-path sam_outputs/data/brain_tumors/test_masks \
    --image-path data/brain_tumors/test_images \
    --output-path evaluation/results/brain_tumors \
    --alpha 0.5 --comparison
```

### 📌 상황 2: 오류를 강하게 강조하고 싶을 때
```bash
python evaluation/visualize_fp_fn.py \
    --gt-path data/breast_tumors/test_masks \
    --pred-path sam_outputs/data/breast_tumors/test_masks \
    --output-path evaluation/results/breast_tumors \
    --alpha 0.8  # 더 진하게
```

### 📌 상황 3: 서로 다른 모델/방법 비교
```bash
# 모델 A (Coarse)
python evaluation/visualize_fp_fn.py \
    --gt-path data/brain_tumors/test_masks \
    --pred-path coarse_outputs/data/brain_tumors/masks \
    --output-path evaluation/results/method_a

# 모델 B (SAM)
python evaluation/visualize_fp_fn.py \
    --gt-path data/brain_tumors/test_masks \
    --pred-path sam_outputs/data/brain_tumors/test_masks \
    --output-path evaluation/results/method_b
```

### 📌 상황 4: 모든 데이터셋 한번에 분석
```bash
bash evaluation/run_full_analysis.sh
```

### 📌 상황 5: 특정 이미지들만 고해상도로 분석
```bash
# 상위 FP 이미지부터 분석
python evaluation/visualize_fp_fn.py \
    --gt-path data/brain_tumors/test_masks \
    --pred-path sam_outputs/data/brain_tumors/test_masks \
    --image-path data/brain_tumors/test_images \
    --output-path evaluation/results/top_fp_analysis \
    --alpha 0.7
```

---

## 📈 성능 평가 표준

### 우수한 모델 기준
```
✅ Precision > 0.90  (오진이 거의 없음)
✅ Recall > 0.90     (병변 검출률이 높음)
✅ F1-Score > 0.90   (전체 성능이 우수함)
✅ 오버레이에서 초록색(TP)이 대부분
```

### 개선 필요한 모델
```
❌ Precision < 0.80  (많은 오진)
❌ Recall < 0.80     (병변을 많이 놓침)
❌ 빨간색(FP)이 많음 (과잉 예측)
❌ 파란색(FN)이 많음 (누락이 많음)
```

---

## 🔧 고급 기능

### 1. 대시보드 생성
```bash
python evaluation/advanced_visualization.py \
    --gt-path data/brain_tumors/test_masks \
    --pred-path sam_outputs/data/brain_tumors/test_masks \
    --image-path data/brain_tumors/test_images \
    --output-path evaluation/results/dashboards \
    --num-images 9  # 한 페이지에 9개 이미지
    --dashboard
```

### 2. 배치 자동화
```bash
# 모든 데이터셋 처리
for dataset in brain_tumors breast_tumors lung_CT lung_Xray; do
    python evaluation/visualize_fp_fn.py \
        --gt-path data/$dataset/test_masks \
        --pred-path sam_outputs/data/$dataset/test_masks \
        --image-path data/$dataset/test_images \
        --output-path evaluation/results/$dataset \
        --alpha 0.5 --comparison
done
```

### 3. 샘플 이미지 생성
```bash
python evaluation/show_samples.py
```

---

## 📂 파일 구조

```
evaluation/
├── visualize_fp_fn.py              ⭐ 메인 스크립트
├── advanced_visualization.py       (대시보드)
├── visualize_examples.sh           (배치 스크립트)
├── run_full_analysis.sh            (전체 분석)
├── show_samples.py                 (샘플 생성)
├── QUICK_START.md                  (빠른 시작)
├── README_VISUALIZATION.md         (상세 가이드)
├── results/                        📁 결과물
│   ├── brain_tumors/
│   │   ├── *.png (오버레이)
│   │   └── comparison/ *.png (비교)
│   ├── breast_tumors/
│   └── lung_CT/
├── dashboards/                     📁 대시보드
│   └── *.png
└── fp_fn_viz/                      📁 테스트 결과
    └── breast_tumors_test/
```

---

## 🎬 실행 흐름도

```
1️⃣ 데이터 준비
   ├─ GT 마스크 (data/*/test_masks/)
   ├─ 예측 마스크 (sam_outputs/data/*/test_masks/)
   └─ 원본 이미지 (data/*/test_images/)
       ↓
2️⃣ 스크립트 실행
   └─ visualize_fp_fn.py
       ↓
3️⃣ 자동 처리
   ├─ 마스크 이진화
   ├─ FP/FN 계산
   ├─ 컬러 오버레이 생성
   └─ 통계 계산
       ↓
4️⃣ 결과물 생성
   ├─ 오버레이 이미지
   ├─ 비교 이미지
   ├─ 통계 정보
   └─ 분석 리포트
       ↓
5️⃣ 분석 및 개선
   ├─ 오류 위치 파악
   ├─ 해부학적 해석
   └─ 개선 방안 수립
```

---

## 🆘 문제 해결

| 문제 | 원인 | 해결책 |
|------|------|--------|
| `ModuleNotFoundError: cv2` | OpenCV 미설치 | `conda activate medclipsamv2` |
| 파일을 찾을 수 없음 | 경로 오류 | `--gt-path`, `--pred-path` 확인 |
| 메모리 부족 | 고해상도 이미지 | 이미지 리사이즈 또는 배치 처리 |
| 결과 이미지가 검은색 | 마스크 형식 오류 | Threshold=127로 이진화 확인 |
| 오버레이가 보이지 않음 | alpha=0 | `--alpha 0.5` 지정 |

---

## 📞 다음 단계

1. **✅ 시각화 실행**
   ```bash
   python evaluation/visualize_fp_fn.py --gt-path ... --pred-path ...
   ```

2. **👁️ 결과 확인**
   - 오버레이 이미지에서 FP/FN 위치 파악
   - 콘솔에서 통계 정보 확인

3. **🔍 상세 분석**
   - 상위 오류 이미지 분석
   - 해부학적 특징 해석
   - 오류 패턴 식별

4. **💡 개선 계획**
   - FP/FN별 해결 전략 수립
   - 학습 데이터 보강 또는 모델 조정
   - Threshold 또는 Post-processing 최적화

5. **🔄 재분석**
   - 개선 후 다시 시각화하여 효과 검증

---

## 🎉 요약

이제 다음이 가능합니다:
- ✅ **시각적 분석**: 모델의 오류를 색깔별로 명확하게 표시
- ✅ **해부학적 이해**: 어느 위치에서 모델이 헷갈리는지 파악
- ✅ **정량적 평가**: Precision, Recall 등 객관적 수치 제공
- ✅ **단계별 비교**: Coarse vs SAM 등 개선 효과 확인
- ✅ **자동화**: 배치 처리로 모든 이미지 일괄 분석

**이 도구를 사용하여 모델의 약점을 파악하고 체계적으로 개선할 수 있습니다! 🚀**

---

## 📚 참고 문서

- [QUICK_START.md](QUICK_START.md) - 5분 빠른 시작
- [README_VISUALIZATION.md](README_VISUALIZATION.md) - 상세 가이드
- [visualize_fp_fn.py](visualize_fp_fn.py) - 메인 스크립트 (주석 포함)
- [advanced_visualization.py](advanced_visualization.py) - 고급 기능
