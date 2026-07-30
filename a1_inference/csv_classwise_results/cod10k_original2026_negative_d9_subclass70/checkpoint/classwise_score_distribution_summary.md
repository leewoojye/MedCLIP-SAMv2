# COD10K GMPO global: macro 평균 이전 class-wise 점수 분포

## 평가 대상

- Checkpoint: `a1_inference/model/cod10k_global/checkpoints/gmpo_global_ind_explicitneg_2886_seed0_epoch_3.pt`
- Backbone: OpenAI CLIP ViT-B/32 구조에 DDP `module.` 접두사를 제거한 가중치를 strict load
- Dataset: COD10K 원본 2,026장 + no-object 음성 샘플 2,026장 = 4,052장
- Classes: 69개 camouflage subclass + `no camouflaged animal` = 70개
- Score: 이미지/텍스트 임베딩을 L2-normalize한 raw cosine similarity
- Metric: 각 클래스 one-vs-rest AUROC 및 ROC 상에서 처음 TPR >= 0.95가 되는 지점의 FPR

`macro` 값은 아래 70개 class-wise 값을 **동일한 비중으로 산술평균**한 값이다. 따라서 양성이 적은 클래스도 `no camouflaged animal`(2,026장)과 동일한 비중을 갖는다.

## 최종 macro 값

| Metric | Value |
|---|---:|
| Macro AUROC | 0.847809 |
| Macro FPR@95%TPR | 0.382762 |

## Macro 이전 class-wise AUROC / FPR 분포

| Metric | Min | P05 | Q1 | Median | Q3 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| AUROC | 0.166831 | 0.557405 | 0.811015 | 0.894668 | 0.950956 | 0.987097 | 0.994366 |
| FPR@95%TPR | 0.015073 | 0.040831 | 0.157103 | 0.351476 | 0.526412 | 0.829059 | 0.979440 |

| AUROC bin | Number of classes |
|---|---:|
| `< 0.50` | 3 |
| `0.50–<0.70` | 5 |
| `0.70–<0.80` | 9 |
| `0.80–<0.90` | 20 |
| `0.90–<0.95` | 14 |
| `>= 0.95` | 19 |

| FPR@95%TPR bin | Number of classes |
|---|---:|
| `< 0.10` | 11 |
| `0.10–<0.30` | 18 |
| `0.30–<0.50` | 21 |
| `0.50–<0.70` | 8 |
| `0.70–<0.90` | 10 |
| `>= 0.90` | 2 |

## Raw cosine score 분포

| Score population | Min | P05 | Q1 | Median | Q3 | P95 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| 전체 이미지-캡션 쌍 | 0.046787 | 0.159596 | 0.199555 | 0.227745 | 0.253222 | 0.283197 | 0.352602 |
| 정답 class caption | 0.153122 | 0.223067 | 0.248121 | 0.266625 | 0.284619 | 0.310620 | 0.352602 |
| 비정답 class caption | 0.046787 | 0.159333 | 0.199105 | 0.227122 | 0.252512 | 0.282328 | 0.352225 |

정답 caption score의 median은 `0.266625`이고 비정답 caption score의 median은 `0.227122`로, 정답 쪽으로의 점수 이동은 확인된다. 다만 두 분포는 상당 부분 겹치므로 일부 클래스에서 높은 FPR이 나타난다.

## 클래스 불균형과 극단값

양성 이미지 수 분포는 median 16장, Q1 7장, Q3 40.5장이다.

| Positive-count bin | Number of classes |
|---|---:|
| `1–5` | 13 |
| `6–10` | 13 |
| `11–25` | 19 |
| `26–100` | 19 |
| `101–500` | 5 |
| `>500` | 1 |

| Case | Class | Positive count | AUROC | FPR@95%TPR |
|---|---|---:|---:|---:|
| Lowest AUROC | pagurian | 6 | 0.166831 | 0.974296 |
| High FPR | other | 15 | 0.389563 | 0.979440 |
| Highest AUROC | leafy sea dragon | 5 | 0.994366 | 0.015073 |
| No-object class | no camouflaged animal | 2,026 | 0.702100 | 0.843534 |

## Source artifacts

- `per_class_metrics.csv`: 모든 70개 class-wise AUROC/FPR 값
- `predictions.npz`: `y_score` raw cosine 행렬 `(4052, 70)` 및 labels
- `metrics.json`: macro 및 ROC metric 요약
- `run_metadata.json`: checkpoint 경로와 평가 설정
