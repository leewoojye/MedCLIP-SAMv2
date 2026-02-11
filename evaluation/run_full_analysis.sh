#!/bin/bash
# ============================================================================
# MedCLIP-SAMv2 모델 예측 결과 시각화 통합 가이드
# FP(False Positive)와 FN(False Negative)를 색깔별로 오버레이
# ============================================================================

echo "============================================================================"
echo "MedCLIP-SAMv2 모델 예측 분석 도구"
echo "============================================================================"
echo ""
echo "목표: 모델의 예측 결과(Mask)와 정답(GT)을 시각적으로 비교"
echo "- FP (빨간색): 모델이 예측했지만 GT에는 없음 (과잉 예측)"
echo "- FN (파란색): GT에는 있지만 모델이 예측하지 못함 (누락)"
echo "- TP (초록색): 모델과 GT가 모두 일치"
echo ""
echo "============================================================================"
echo ""

# 색상 설정
RED='\033[0;31m'
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 환경 활성화
echo -e "${YELLOW}[Step 1] conda 환경 활성화...${NC}"
eval "$(conda shell.bash hook)"
conda activate medclipsamv2

# 프로젝트 디렉토리 확인
PROJECT_DIR="/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2"
cd "$PROJECT_DIR"

echo -e "${GREEN}✓ 환경 준비 완료${NC}\n"

# 1. 단일 데이터셋 분석
echo "============================================================================"
echo "방법 1: 특정 데이터셋 분석하기"
echo "============================================================================"
echo ""
echo "사용 가능한 데이터셋:"
ls -d data/*/ 2>/dev/null | xargs -I {} basename {}
echo ""

# 뇌종양 분석
echo -e "${YELLOW}[분석 1] 뇌종양 - 기본 설정${NC}"
python evaluation/visualize_fp_fn.py \
    --gt-path data/brain_tumors/test_masks \
    --pred-path sam_outputs/data/brain_tumors/test_masks \
    --image-path data/brain_tumors/test_images \
    --output-path evaluation/results/brain_tumors \
    --alpha 0.5 \
    --comparison

echo ""
echo -e "${YELLOW}[분석 2] 유방종양 - 강한 오버레이 (오류 강조)${NC}"
python evaluation/visualize_fp_fn.py \
    --gt-path data/breast_tumors/test_masks \
    --pred-path sam_outputs/data/breast_tumors/test_masks \
    --image-path data/breast_tumors/test_images \
    --output-path evaluation/results/breast_tumors \
    --alpha 0.7 \
    --comparison

echo ""
echo -e "${YELLOW}[분석 3] 폐 CT - 투명한 오버레이 (원본 이미지 중시)${NC}"
python evaluation/visualize_fp_fn.py \
    --gt-path data/lung_CT/test_masks \
    --pred-path sam_outputs/data/lung_CT/test_masks \
    --image-path data/lung_CT/test_images \
    --output-path evaluation/results/lung_CT \
    --alpha 0.3 \
    --comparison

echo ""

# 2. 처리 단계별 비교
echo "============================================================================"
echo "방법 2: 처리 단계별 비교 (Coarse vs SAM)"
echo "============================================================================"
echo ""

echo -e "${YELLOW}Coarse 출력 vs SAM 최종 결과 비교${NC}"
echo ""

# Coarse 단계
echo "Coarse 단계 시각화..."
python evaluation/visualize_fp_fn.py \
    --gt-path data/brain_tumors/test_masks \
    --pred-path coarse_outputs/data/brain_tumors/masks \
    --image-path data/brain_tumors/test_images \
    --output-path evaluation/results/comparison/coarse \
    --alpha 0.5

# SAM 단계
echo ""
echo "SAM 단계 시각화..."
python evaluation/visualize_fp_fn.py \
    --gt-path data/brain_tumors/test_masks \
    --pred-path sam_outputs/data/brain_tumors/test_masks \
    --image-path data/brain_tumors/test_images \
    --output-path evaluation/results/comparison/sam \
    --alpha 0.5

echo ""
echo -e "${GREEN}✓ Coarse와 SAM의 개선 효과를 비교할 수 있습니다${NC}"
echo ""

# 3. 결과 요약
echo "============================================================================"
echo "결과 저장 위치"
echo "============================================================================"
echo ""
echo "오버레이 이미지: evaluation/results/"
echo "비교 이미지:    evaluation/results/*/comparison/"
echo ""

# 4. 통계 분석
echo "============================================================================"
echo "모델 성능 분석 팁"
echo "============================================================================"
echo ""
echo "🔴 FP (False Positive, 빨간색) 분석:"
echo "  - 위치: 정상 조직 또는 경계 영역에 많이 나타남"
echo "  - 의미: 모델이 보수적이지 못함 (민감도 높음)"
echo "  - 해결: Threshold 조정 또는 Post-processing으로 작은 영역 제거"
echo ""
echo "🔵 FN (False Negative, 파란색) 분석:"
echo "  - 위치: 병변의 중심 또는 가장자리에 나타남"
echo "  - 의미: 모델이 병변을 놓침 (재현율 낮음)"
echo "  - 해결: 학습 데이터 보강 또는 모델 미세 조정"
echo ""
echo "🟢 TP (True Positive, 초록색):"
echo "  - 올바르게 예측된 영역"
echo "  - 이 영역이 많을수록 모델이 잘 작동"
echo ""

# 5. 고급 기능
echo "============================================================================"
echo "고급 기능: 대시보드 시각화"
echo "============================================================================"
echo ""
echo -e "${YELLOW}여러 이미지를 한 페이지에 격자로 표시${NC}"
echo ""

python evaluation/advanced_visualization.py \
    --gt-path data/brain_tumors/test_masks \
    --pred-path sam_outputs/data/brain_tumors/test_masks \
    --image-path data/brain_tumors/test_images \
    --output-path evaluation/results/dashboards \
    --num-images 6 \
    --dashboard

echo ""
echo -e "${GREEN}✓ 대시보드가 생성되었습니다: evaluation/results/dashboards/${NC}"
echo ""

# 최종 요약
echo "============================================================================"
echo "✅ 분석 완료!"
echo "============================================================================"
echo ""
echo "생성된 결과물:"
echo "  1. 오버레이 이미지 - FP/FN을 색깔별로 표시한 이미지"
echo "  2. 비교 이미지 - 원본 / 예측 / GT를 나란히 비교"
echo "  3. 통계 정보 - Precision, Recall, F1-Score"
echo "  4. 대시보드 - 여러 이미지를 한 페이지에 표시"
echo ""
echo "다음 단계:"
echo "  - evaluation/results/ 폴더에서 이미지 확인"
echo "  - 상위 FP/FN 이미지 분석"
echo "  - 해부학적 위치별 오류 패턴 파악"
echo "  - 모델 개선 방안 수립"
echo ""
echo "============================================================================"
