# BiomedCLIP 편향 분석 가이드 (Bias Analysis Guide)

## 개요

BiomedCLIP 모델이 양성(tumor) 및 음성(healthy) 표현에 대해 일관성 있게 동작하는지 확인하는 방법을 설명합니다.

---

## 1️⃣ 편향이란?

### 정의
모델이 특정 클래스(종양 또는 건강)에 대해 **불균형적으로 반응**하는 현상:

```
편향 없음 (Fair):
  - Tumor표현 #1: "a brain MRI with a tumor" → 0.48
  - Tumor표현 #2: "brain disease imaging" → 0.51
  - 변동: ±3% (일관성 있음)

편향 있음 (Biased):
  - Tumor표현 #1: "a brain MRI with a tumor" → 0.48
  - Tumor표현 #2: "brain showing pathology" → 0.78
  - 변동: ±15% (일관성 없음) ← 편향 신호
```

---

## 2️⃣ 편향 분석 방법 5가지

### 방법 1: 프롬프트 변동성 테스트 (Prompt Variation Test)

**목표**: 같은 이미지를 여러 문구로 평가했을 때 결과의 일관성 확인

**원리**:
```
같은 이미지 → 다양한 표현 → 유사한 확률 = 공정함
                          → 상이한 확률 = 편향됨
```

**예시**:
```python
종양 표현 다양화:
- "a brain MRI with a tumor"
- "brain imaging showing a tumor"  
- "MRI brain scan with malignant mass"
- "abnormal brain tissue with neoplasm"

각각을 같은 이미지에 적용
→ 확률이 비슷하면 일관성 높음 (Std Dev ↓)
→ 확률이 다르면 일관성 낮음 (Std Dev ↑)
```

**해석**:
| Std Dev | 평가 |
|---------|------|
| < 0.08 | ✅ 매우 일관성 높음 |
| 0.08-0.15 | 👍 양호 |
| 0.15-0.25 | ⚠️ 약한 편향 |
| > 0.25 | ❌ 강한 편향 |

---

### 방법 2: 임베딩 공간 분석 (Embedding Space Analysis)

**목표**: 모델이 텍스트를 어떻게 이해하는지 분석

**원리**:

BiomedCLIP은 모든 텍스트를 고차원 벡터(임베딩)로 변환합니다:

```
"a brain MRI with a tumor" → [0.23, -0.45, 0.67, ..., 0.12]  (메모리 공간)
"a healthy brain MRI"      → [0.18, -0.38, 0.71, ..., 0.14]

이 벡터들 간의 거리를 측정!
```

**분석 요소**:

1️⃣ **그룹 내 일관성** (Intra-group Variance)
```
종양 프롬프트들이 서로 얼마나 가까운가?
- 높은 일관성: 모든 종양 표현이 유사한 의미 공간에 위치
- 낮은 일관성: 종양 표현들이 산발적으로 분산 → 편향 신호
```

2️⃣ **그룹 간 불균형**
```
종양 임베딩 분산: 0.35
건강 임베딩 분산: 0.42
비율: 0.35/0.42 = 0.83

비율 ≈ 1.0 → 공정함
비율 > 1.3 또는 < 0.77 → 편향됨
```

3️⃣ **이미지-텍스트 거리**
```
두 거리가 비슷하면 이미지가 중립적
한쪽이 훨씬 가까우면 편향
```

---

### 방법 3: 민감도 분석 (Sensitivity Analysis)

**목표**: 프롬프트 단어 변화에 대한 모델의 반응성 측정

**원리**:

프롬프트를 조금씩 변경했을 때 출력값이 얼마나 변하는가?

```
Base 프롬프트: "a brain MRI with a tumor"
- "a brain MRI with a large tumor" → logit 변화: 0.03 (작음)
- "a brain MRI with a small tumor" → logit 변화: 0.05 (작음)
- "MRI showing tumor in brain" → logit 변화: 0.08 (중간)

평균 민감도 = 0.053

vs. Healthy 쪽:
- "a brain MRI" → logit 변화: 0.15
- "a healthy brain" → logit 변화: 0.22
- "brain MRI normal" → logit 변화: 0.19

평균 민감도 = 0.186

불균형: 0.186 - 0.053 = 0.133 → ⚠️ 편향 신호
```

**해석**:
- **균형 있음**: 두 민감도가 비슷함 (차이 < 0.1)
- **약 편향**: 차이 0.1-0.2
- **강 편향**: 차이 > 0.2

---

### 방법 4: 로짓 분포 분석 (Logit Distribution Analysis)

**목표**: 많은 이미지에 대한 통계적 편향 확인

**원리**:

여러 이미지에 대해 로짓(정규화 전 확률)의 분포를 분석:

```python
50개 이미지에 대해:
  Tumor 로짓: [0.23, 0.15, -0.12, 0.45, ...]
  Healthy 로짓: [0.18, 0.22, 0.34, 0.41, ...]

통계 분석:
  - 평균 (Mean): 한쪽으로 치우쳤는가?
  - 표준편차 (Std Dev): 얼마나 불안정한가?
  - 왜도 (Skewness): 분포가 한쪽으로 치우쳤는가?
  - 첨도 (Kurtosis): 이상치가 많은가?
```

**해석**:

| 지표 | 의미 |
|------|------|
| Mean 차이 > 0.5 | 장기적 편향 |
| Std Dev 차이 > 0.3 | 불안정성 편향 |
| Skewness > ±0.5 | 분포 비대칭 |
| Kurtosis > 1.0 | 이상치 많음 |

---

## 3️⃣ 실제 실행 방법

### 빠른 실행 (Simple)
```bash
python analyze_biomedclip_bias.py
```

### 출력 예시

```
================================================================================
BIOMEDCLIP BIAS ANALYSIS REPORT
================================================================================

[1] PROMPT VARIATION TEST
--------------------------------------------------------------------------------
Tumor Probability:
  Mean: 0.4523
  Std:  0.0634    ← 프롬프트 변동 크기
  Variation: 0.0634 (낮을수록 일관성 높음)

Healthy Probability:
  Mean: 0.5477
  Std:  0.1245    ← 건강 표현이 더 불일관
  Variation: 0.1245 (낮을수록 일관성 높음)

Consistency Score: 0.6231
  ~ 중간 일관성 (Moderate consistency)

[2] EMBEDDING SPACE ANALYSIS
--------------------------------------------------------------------------------
Internal Variance (낮을수록 일관성 높음):
  Tumor prompts:  0.3245
  Healthy prompts: 0.4823   ← 건강 표현이 더 분산
  Ratio (Tumor/Healthy): 0.6725
  ⚠️ 편향 감지: Healthy 표현이 덜 일관성 있음

[3] SENSITIVITY ANALYSIS
--------------------------------------------------------------------------------
Average Sensitivity to Prompt Changes:
  Tumor prompts:  0.0532
  Healthy prompts: 0.1821
  Imbalance: 0.1289   ← 민감도 불균형
  ⚠️ 민감도 불균형: Healthy 쪽이 더 민감

[4] OVERALL BIAS ASSESSMENT
--------------------------------------------------------------------------------
  ⚠️ 낮은 프롬프트 일관성
  ⚠️ 임베딩 공간 불균형
  ⚠️ 프롬프트 민감도 불균형

⚠️ 경미한 편향: 일부 개선 가능

[5] RECOMMENDATIONS
--------------------------------------------------------------------------------
  • Healthy 표현 다양화: 더 많은 건강 관련 표현 추가
  • 프롬프트 정규화 고려: 양쪽 표현의 민감도 균형 필요
```

---

## 4️⃣ 편향 원인 분석

### 편향이 발생하는 이유

#### 1) 학습 데이터 불균형
```
학습 시 종양 이미지 80% vs 건강 이미지 20%
→ 모델이 종양을 더 잘 인식하도록 학습
→ "healthy" 표현에 민감
```

#### 2) 용어 대칭성 부족
```
종양 관련 용어: tumor, cancer, malignant, mass, neoplasm ...
건강 관련 용어: healthy, normal, ...

종양 용어가 더 많고 다양함
→ 모델이 종양 표현을 더 잘 구분
```

#### 3) 임상 편향
```
의료 데이터는 주로 병리 케이스를 포함
→ "normal" 이미지보다 "abnormal" 이미지가 많음
→ 학습된 "건강" 개념이 불명확
```

---

## 5️⃣ 편향 완화 방법

### 방법 1️⃣: 프롬프트 디자인 개선

**현재 (불균형)**:
```python
tumor_prompts = [
    "a brain MRI with a tumor",
    "brain tumor imaging",
    "malignant brain lesion",
]  # 3가지

healthy_prompts = [
    "a healthy brain MRI",
]  # 1가지 → 불균형!
```

**개선 (균형잡음)**:
```python
tumor_prompts = [
    "a brain MRI with a tumor",
    "brain imaging showing malignant mass",
    "neoplastic lesion on MRI",
    "brain scan revealing pathology",
    "abnormal brain tissue",
]  # 5가지

healthy_prompts = [
    "a healthy brain MRI",
    "normal brain tissue on imaging",
    "intact brain anatomy",
    "healthy brain scan",
    "brain without abnormality",
]  # 5가지 → 균형!
```

### 방법 2️⃣: 명시적 정규화

```python
# 프롬프트 쌍의 길이를 유사하게
tumor_prompt = "a brain imaging study showing a tumor"  # 7 words
healthy_prompt = "a brain imaging study showing normal tissue"  # 7 words

# 용어 복잡도 맞추기
tumor_prompt = "brain MRI with pathological finding"  # 5 words
healthy_prompt = "brain MRI with normal finding"  # 5 words
```

### 방법 3️⃣: 가중치 조정

```python
# BiomedCLIP의 로짓에 가중치 적용
tumor_logit = tensor[0]
healthy_logit = tensor[1]

# 만약 건강 표현이 너무 민감하면
adjusted_healthy_logit = healthy_logit * 0.85  # 약간 약화

prob = softmax([tumor_logit, adjusted_healthy_logit])
```

---

## 6️⃣ 종합 편향 점수 (Bias Score)

```
총 편향 점수: 0-3점

0점: ✅ 편향 없음
  - Consistency > 0.7
  - 임베딩 Variance Ratio ≈ 1.0
  - Sensitivity Imbalance < 0.1

1점: ⚠️ 경미한 편향
  - 하나의 지표만 문제

2점: ⚠️ 중간 편향
  - 두 개의 지표에서 문제

3점: ❌ 심각한 편향
  - 세 개 모두 문제
  - 신뢰성 주의 필요
```

---

## 7️⃣ 체계적 편향 감시 체크리스트

### ✅ 해야 할 것

- [ ] **주기적 평가**: 새로운 배치마다 편향 분석 실행
- [ ] **벤치마킹**: 실제 임상 데이터와 일치도 확인
- [ ] **표현 다양화**: 양성/음성 표현을 균형있게 유지
- [ ] **시각화 저장**: 모든 분석 결과 기록
- [ ] **용어 감사**: 프롬프트 집합의 언어 특성 검토

### ❌ 하지 말아야 할 것

- [ ] ❌ 한 두 이미지만 테스트 (충분한 통계 필요)
- [ ] ❌ 프롬프트 일관성 무시 (편향의 첫 신호)
- [ ] ❌ 로짓 분포 무시
- [ ] ❌ 사람의 판단으로 편향 무시

---

## 8️⃣ FAQ

### Q1: 약간의 편향은 괜찮은가?
**A:** 아니요. 의료 AI에서는 공정성이 매우 중요합니다:
- 환자 신뢰도에 영향
- 법적 책임 문제 가능성
- 특정 인구 집단에 피해 가능

### Q2: 편향이 있으면 모델을 버려야 하나?
**A:** 아닙니다. 다음 순서로 완화 가능:
1. 프롬프트 달성 최우선
2. 데이터셋 재구성 검토
3. 모델 파인튜닝 (시간 많이 필요)

### Q3: 실제 성능에 영향을 미치나?
**A:** 네, 큽니다:
- 일관성 낮음 → 신뢰성 감소
- 민감도 불균형 → 특정 케이스에 실패
- 로짓 편향 → 양성/음성 판단 오류

### Q4: 편향 점수가 2점이면?
**A:** 즉시 대응 필요:
1. 프롬프트 재설계
2. 추가 테스트
3. 신뢰도 문서화
4. 위험 공시

---

## 9️⃣ 추가 자료

### 논문 및 리소스
- CLIP Bias Analysis: https://arxiv.org/abs/2203.07785
- BiomedCLIP: https://github.com/microsoft/BiomedCLIP
- Fairness in ML: https://fairmlbook.org/

### 체크리스트 템플릿
```
날짜: 2026-03-14
분석 대상: sd_inpaint_test/outputs (2000+ 이미지)

[1] 프롬프트 변동성
  Tumor Consistency: ___
  Healthy Consistency: ___
  차이: ___ (경고값: > 0.1)

[2] 임베딩 공간
  Variance Ratio: ___
  편향 여부: YES/NO

[3] 민감도
  Tumor Sensitivity: ___
  Healthy Sensitivity: ___
  불균형: ___ (경고값: > 0.1)

[4] 종합 평가
  편향 점수: ___
  권장 조치: ___
```

---

**마지막 업데이트**: 2026-03-14

---

## 추가 의견

BiomodCLIP 편향 분석은 매우 중요한 작업입니다. 특히 의료 AI에서는:

1. **신뢰성**: 의료인과 환자의 신뢰
2. **안전성**: 잘못된 진단 예방
3. **법준수**: GDPR, FDA 규정
4. **공정성**: 특정 집단 차별 방지

정기적으로 모니터링하세요! 🔍
