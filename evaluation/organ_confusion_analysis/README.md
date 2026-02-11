# 췌장 분할 모델 - 장기별 오인율 분석

## 📋 개요

이 분석은 MedCLIP-SAMv2 모델이 췌장 분할 시 다른 장기들을 얼마나 오인하는지 정량적으로 측정합니다.

## 🔍 주요 발견사항

### 장기별 오인율 (전체 FP 대비)

| 순위 | 장기 | FP 비율 | 영향받은 케이스 |
|------|------|---------|-----------------|
| 1 | 우측 신장 | 43.1% | 120/281 (42.7%) |
| 2 | 간 | 26.4% | 99/281 (35.2%) |
| 3 | 십이지장 | 11.5% | 34/281 (12.1%) |
| 4 | 대혈관 | 9.4% | 26/281 (9.3%) |
| 5 | 좌측 신장 | 5.9% | 19/281 (6.8%) |

**핵심 결과**:
- 🔴 **신장(양측) = 49%** - 가장 심각한 문제
- 🔴 **간 = 26%** - 두 번째 문제
- ✅ **상위 3개 장기가 전체 FP의 81%를 차지**

## 📊 생성된 파일

### 1. 보고서
- [`organ_confusion_report.md`](organ_confusion_analysis/organ_confusion_report.md) - 상세 분석 보고서

### 2. 데이터
- [`organ_confusion_summary.json`](organ_confusion_analysis/organ_confusion_summary.json) - 장기별 통계 요약
- [`detailed_results.json`](organ_confusion_analysis/detailed_results.json) - 케이스별 상세 결과

### 3. 시각화
- `fp_pixels_by_organ.png` - 장기별 FP 픽셀 수
- `fp_distribution_pie.png` - FP 분포 파이 차트
- `cases_affected_by_organ.png` - 장기별 영향받은 케이스 수
- `avg_fp_per_case.png` - 장기별 케이스당 평균 FP
- `confusion_dashboard.png` - 통합 대시보드

## 🚀 실행 방법

### 1. 장기별 오인율 분석 실행
```bash
cd /home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2
python evaluation/analyze_organ_confusion.py
```

### 2. 시각화 생성
```bash
python evaluation/visualize_organ_confusion.py
```

## 📁 필요한 데이터

분석에 필요한 파일들:
- `data/pancreas/test_images/` - 원본 CT 이미지
- `data/pancreas/test_masks/` - Ground truth 마스크
- `sam_outputs/data/pancreas/test_masks/` - 모델 예측 마스크

## 🔬 분석 방법론

### 공간적 위치 기반 장기 분류
CT 복부 영상의 표준 해부학적 위치를 기반으로 FP(False Positive) 영역을 분류:

1. **이미지 구역 정의**: 각 장기의 전형적인 위치 영역 정의
2. **FP 영역 추출**: 예측 마스크에서 FP 픽셀 추출
3. **연결 성분 분석**: FP를 개별 컴포넌트로 분리
4. **위치 매칭**: 각 컴포넌트가 어떤 장기 영역과 겹치는지 계산
5. **통계 집계**: 장기별 FP 픽셀 수 및 케이스 수 집계

### 장기 위치 정의 예시
```python
organ_regions = {
    'liver': (y: 0.2-0.6, x: 0.3-0.8),      # 우상복부
    'spleen': (y: 0.25-0.55, x: 0.05-0.35), # 좌상복부
    'pancreas': (y: 0.35-0.55, x: 0.30-0.60), # 중앙
    'kidney_right': (y: 0.35-0.65, x: 0.55-0.80), # 우후복부
    # ... 기타 장기
}
```

## 💡 개선 권장사항

### 1. Text Prompt 개선
```python
# 현재
prompt = "pancreas"

# 개선안 (우선순위 순)
prompt = "pancreas only, exclude kidney and liver"
prompt = "pancreas only, not kidney, not liver, not duodenum"
```

### 2. 공간적 후처리
```python
# 신장 영역 제거
- 우측 신장: x > 0.55, y: 0.35-0.65
- 좌측 신장: x < 0.35, y: 0.35-0.65

# 간 영역 제거
- 우상부: y < 0.35, x > 0.50

# 췌장 영역만 유지
- 중앙부: y: 0.35-0.55, x: 0.30-0.60
```

### 3. 크기 제약
```python
# 췌장의 전형적인 크기 범위
pancreas_size_range = (0.5%, 3.0%) of image area

if predicted_area > 3% of image:
    apply_size_reduction()
```

## 📈 통계 요약

- **총 분석 케이스**: 281개
- **총 FP 픽셀**: 3,172,917개
- **평균 Dice Score**: 0.046
- **신장+간의 FP 기여도**: 69.5%

## 🔗 관련 파일

- [`PANCREAS_WEAKNESS_ANALYSIS.md`](PANCREAS_WEAKNESS_ANALYSIS.md) - 전체 약점 분석
- [`analyze_organ_confusion.py`](analyze_organ_confusion.py) - 분석 스크립트
- [`visualize_organ_confusion.py`](visualize_organ_confusion.py) - 시각화 스크립트

## 📝 업데이트 로그

- **2024-12-29**: 초기 분석 완료
  - 281개 케이스 분석
  - 7개 주요 장기 식별
  - 신장이 가장 큰 혼동 대상임을 발견

## 🎯 향후 작업

- [ ] 실제 개선 방안 구현 및 테스트
- [ ] 신장 배제 prompt 효과 검증
- [ ] 공간적 후처리 알고리즘 개발
- [ ] Few-shot fine-tuning으로 성능 개선
