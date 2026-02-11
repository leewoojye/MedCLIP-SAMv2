"""
췌장 분할에서 장기별 오인율 분석
FP 영역의 공간적 위치와 특성을 분석하여 어떤 장기로 오인했는지 추정
"""

import numpy as np
from PIL import Image
import os
from pathlib import Path
from collections import defaultdict
import json
from scipy import ndimage
import matplotlib.pyplot as plt

class OrganRegionAnalyzer:
    """
    CT 영상에서 복부 장기들의 전형적인 위치를 기반으로 
    FP 영역이 어떤 장기에 해당하는지 추정
    """
    
    def __init__(self, image_shape=(512, 512)):
        """
        췌장과 인접 장기들의 전형적인 위치 정의 (정규화된 좌표)
        CT 복부 영상의 표준 해부학적 위치 기반
        """
        h, w = image_shape
        center_y, center_x = h // 2, w // 2
        
        # 각 장기의 전형적인 위치 영역 (y_min, y_max, x_min, x_max)
        # 정규화된 좌표 (0.0 ~ 1.0)
        self.organ_regions = {
            'liver': {
                'region': (0.2, 0.6, 0.3, 0.8),  # 오른쪽 위, 큰 영역
                'priority': 1,  # 가장 큰 장기이므로 우선순위 높음
                'typical_size_range': (0.15, 0.40),  # 이미지 대비 비율
                'description': '간 - 우상복부의 가장 큰 장기'
            },
            'spleen': {
                'region': (0.25, 0.55, 0.05, 0.35),  # 왼쪽 위
                'priority': 2,
                'typical_size_range': (0.03, 0.12),
                'description': '비장 - 좌상복부, 췌장 왼쪽'
            },
            'stomach': {
                'region': (0.25, 0.50, 0.20, 0.50),  # 중앙 상부
                'priority': 3,
                'typical_size_range': (0.05, 0.20),
                'description': '위 - 중앙 상부, 췌장 앞쪽'
            },
            'pancreas': {
                'region': (0.35, 0.55, 0.30, 0.60),  # 중앙
                'priority': 0,  # 목표 장기
                'typical_size_range': (0.005, 0.03),  # 작은 장기
                'description': '췌장 - 중앙 후복부, 작고 길쭉한 형태'
            },
            'duodenum': {
                'region': (0.40, 0.60, 0.40, 0.65),  # 췌장 오른쪽
                'priority': 4,
                'typical_size_range': (0.01, 0.05),
                'description': '십이지장 - 췌장 두부와 인접'
            },
            'kidney_right': {
                'region': (0.35, 0.65, 0.55, 0.80),  # 오른쪽 후복부
                'priority': 5,
                'typical_size_range': (0.02, 0.08),
                'description': '우측 신장 - 후복부'
            },
            'kidney_left': {
                'region': (0.35, 0.65, 0.15, 0.40),  # 왼쪽 후복부
                'priority': 5,
                'typical_size_range': (0.02, 0.08),
                'description': '좌측 신장 - 후복부'
            },
            'vessels': {
                'region': (0.30, 0.60, 0.35, 0.55),  # 중앙, 척추 앞
                'priority': 6,
                'typical_size_range': (0.01, 0.05),
                'description': '대혈관 (대동맥, 하대정맥)'
            }
        }
        
        self.image_shape = image_shape
    
    def normalize_coordinates(self, y, x):
        """좌표를 0~1로 정규화"""
        h, w = self.image_shape
        return y / h, x / w
    
    def point_in_region(self, y_norm, x_norm, region):
        """정규화된 좌표가 특정 영역 안에 있는지 확인"""
        y_min, y_max, x_min, x_max = region
        return (y_min <= y_norm <= y_max) and (x_min <= x_norm <= x_max)
    
    def calculate_overlap_score(self, mask_coords_norm, region):
        """마스크와 장기 영역의 겹침 점수 계산"""
        y_min, y_max, x_min, x_max = region
        
        overlap_count = 0
        for y_norm, x_norm in mask_coords_norm:
            if self.point_in_region(y_norm, x_norm, region):
                overlap_count += 1
        
        return overlap_count / len(mask_coords_norm) if len(mask_coords_norm) > 0 else 0
    
    def analyze_mask_position(self, mask):
        """
        마스크의 위치를 분석하여 가장 가능성 높은 장기 추정
        Returns: dict with organ probabilities
        """
        if np.sum(mask) == 0:
            return {}
        
        # 마스크의 모든 픽셀 좌표 추출
        y_coords, x_coords = np.where(mask)
        
        # 정규화된 좌표
        coords_norm = [(y / self.image_shape[0], x / self.image_shape[1]) 
                       for y, x in zip(y_coords, x_coords)]
        
        # 각 장기와의 겹침 계산
        organ_scores = {}
        for organ_name, organ_info in self.organ_regions.items():
            overlap_score = self.calculate_overlap_score(coords_norm, organ_info['region'])
            
            if overlap_score > 0:
                # 크기 적합성 점수
                mask_size_ratio = np.sum(mask) / (self.image_shape[0] * self.image_shape[1])
                size_min, size_max = organ_info['typical_size_range']
                
                if size_min <= mask_size_ratio <= size_max * 3:  # 3배까지 허용
                    size_score = 1.0
                elif mask_size_ratio > size_max * 3:
                    # 너무 크면 큰 장기일 가능성
                    size_score = 0.5 if organ_name == 'liver' else 0.2
                else:
                    size_score = 0.5
                
                # 최종 점수
                organ_scores[organ_name] = overlap_score * size_score
        
        return organ_scores
    
    def classify_fp_region(self, fp_mask, gt_mask=None):
        """
        FP 영역을 분석하여 가장 가능성 높은 장기 분류
        """
        organ_scores = self.analyze_mask_position(fp_mask)
        
        if not organ_scores:
            return 'unknown', 0.0, {}
        
        # 췌장 제외
        if 'pancreas' in organ_scores:
            del organ_scores['pancreas']
        
        if not organ_scores:
            return 'other', 0.0, {}
        
        # 가장 높은 점수의 장기 선택
        best_organ = max(organ_scores.items(), key=lambda x: x[1])
        
        return best_organ[0], best_organ[1], organ_scores


def analyze_single_case(image_path, pred_mask_path, gt_mask_path, analyzer):
    """개별 케이스 분석"""
    try:
        # 이미지 로드
        image = np.array(Image.open(image_path).convert('L'))
        pred_mask = np.array(Image.open(pred_mask_path).convert('L'))
        gt_mask = np.array(Image.open(gt_mask_path).convert('L'))
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None
    
    # 이미지 크기 업데이트
    analyzer.image_shape = image.shape
    
    # 바이너리 마스크로 변환
    pred_bin = (pred_mask > 127).astype(np.uint8)
    gt_bin = (gt_mask > 127).astype(np.uint8)
    
    # FP, FN, TP 계산
    fp_mask = (pred_bin == 1) & (gt_bin == 0)
    fn_mask = (pred_bin == 0) & (gt_bin == 1)
    tp_mask = (pred_bin == 1) & (gt_bin == 1)
    
    fp_count = np.sum(fp_mask)
    fn_count = np.sum(fn_mask)
    tp_count = np.sum(tp_mask)
    
    # Dice 계산
    dice = 2 * tp_count / (2 * tp_count + fp_count + fn_count) if (2 * tp_count + fp_count + fn_count) > 0 else 0
    
    result = {
        'dice': float(dice),
        'fp_count': int(fp_count),
        'fn_count': int(fn_count),
        'tp_count': int(tp_count),
        'pred_area': int(np.sum(pred_bin)),
        'gt_area': int(np.sum(gt_bin)),
        'confused_organs': {}
    }
    
    # FP 영역 분석
    if fp_count > 0:
        # 연결된 컴포넌트로 분리
        labeled_fp, num_components = ndimage.label(fp_mask)
        
        # 각 컴포넌트를 개별적으로 분석
        organ_fp_counts = defaultdict(int)
        organ_confidence = defaultdict(list)
        
        for component_id in range(1, num_components + 1):
            component_mask = (labeled_fp == component_id)
            component_size = np.sum(component_mask)
            
            # 너무 작은 컴포넌트는 무시 (노이즈)
            if component_size < 10:
                continue
            
            # 장기 분류
            organ_name, confidence, all_scores = analyzer.classify_fp_region(component_mask, gt_bin)
            
            if confidence > 0.1:  # 최소 신뢰도
                organ_fp_counts[organ_name] += component_size
                organ_confidence[organ_name].append(confidence)
        
        # 결과 정리
        for organ_name, count in organ_fp_counts.items():
            avg_confidence = np.mean(organ_confidence[organ_name])
            result['confused_organs'][organ_name] = {
                'fp_pixels': int(count),
                'percentage': float(count / fp_count * 100),
                'confidence': float(avg_confidence)
            }
    
    return result


def main():
    base_dir = Path('/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2')
    
    # 데이터 경로
    test_images_dir = base_dir / 'data' / 'pancreas' / 'test_images'
    test_masks_dir = base_dir / 'data' / 'pancreas' / 'test_masks'
    sam_outputs_dir = base_dir / 'sam_outputs' / 'data' / 'pancreas' / 'test_masks'
    
    # 출력 디렉토리
    output_dir = base_dir / 'evaluation' / 'organ_confusion_analysis'
    output_dir.mkdir(exist_ok=True)
    
    # Analyzer 초기화
    analyzer = OrganRegionAnalyzer()
    
    # 모든 케이스 분석
    all_results = {}
    organ_total_fp = defaultdict(int)
    organ_case_count = defaultdict(int)
    total_fp = 0
    
    # 이미지 파일 리스트
    image_files = sorted([f for f in os.listdir(test_images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Analyzing {len(image_files)} cases...")
    
    for img_file in image_files:
        case_id = img_file.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        
        image_path = test_images_dir / img_file
        gt_mask_path = test_masks_dir / img_file
        pred_mask_path = sam_outputs_dir / img_file
        
        if not pred_mask_path.exists():
            continue
        
        # 분석
        result = analyze_single_case(str(image_path), str(pred_mask_path), 
                                     str(gt_mask_path), analyzer)
        
        if result is None:
            continue
        
        all_results[case_id] = result
        
        # 통계 누적
        total_fp += result['fp_count']
        for organ_name, organ_data in result['confused_organs'].items():
            organ_total_fp[organ_name] += organ_data['fp_pixels']
            organ_case_count[organ_name] += 1
    
    print(f"\nAnalyzed {len(all_results)} cases successfully.")
    
    # 전체 통계 계산
    summary = {
        'total_cases': len(all_results),
        'total_fp_pixels': int(total_fp),
        'organ_confusion_stats': {}
    }
    
    # 장기별 통계
    for organ_name in sorted(organ_total_fp.keys(), key=lambda x: organ_total_fp[x], reverse=True):
        fp_count = organ_total_fp[organ_name]
        case_count = organ_case_count[organ_name]
        
        summary['organ_confusion_stats'][organ_name] = {
            'total_fp_pixels': int(fp_count),
            'percentage_of_total_fp': float(fp_count / total_fp * 100) if total_fp > 0 else 0,
            'cases_affected': int(case_count),
            'percentage_of_cases': float(case_count / len(all_results) * 100),
            'avg_fp_per_case': float(fp_count / case_count) if case_count > 0 else 0,
            'description': analyzer.organ_regions.get(organ_name, {}).get('description', 'Unknown')
        }
    
    # 결과 저장
    with open(output_dir / 'organ_confusion_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    with open(output_dir / 'detailed_results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    # 보고서 생성
    generate_report(summary, all_results, output_dir, analyzer)
    
    print(f"\n결과가 저장되었습니다: {output_dir}")
    print(f"  - organ_confusion_summary.json")
    print(f"  - detailed_results.json")
    print(f"  - organ_confusion_report.md")


def generate_report(summary, all_results, output_dir, analyzer):
    """분석 결과 보고서 생성"""
    
    report = []
    report.append("# 췌장 분할 모델의 장기별 오인율 분석 보고서\n")
    report.append(f"분석 일자: {Path(__file__).stat().st_mtime}\n")
    report.append(f"총 분석 케이스: {summary['total_cases']}개\n")
    report.append(f"총 FP 픽셀: {summary['total_fp_pixels']:,}개\n\n")
    
    report.append("## 📊 장기별 오인 통계\n\n")
    report.append("### 요약 테이블\n\n")
    report.append("| 순위 | 장기 | FP 픽셀 수 | 전체 FP 대비 | 영향받은 케이스 | 케이스당 평균 FP |\n")
    report.append("|------|------|------------|--------------|-----------------|------------------|\n")
    
    organ_stats = summary['organ_confusion_stats']
    sorted_organs = sorted(organ_stats.items(), 
                          key=lambda x: x[1]['total_fp_pixels'], 
                          reverse=True)
    
    for rank, (organ_name, stats) in enumerate(sorted_organs, 1):
        report.append(
            f"| {rank} | **{organ_name.upper()}** | "
            f"{stats['total_fp_pixels']:,} | "
            f"{stats['percentage_of_total_fp']:.1f}% | "
            f"{stats['cases_affected']}/{summary['total_cases']} ({stats['percentage_of_cases']:.1f}%) | "
            f"{stats['avg_fp_per_case']:.0f} |\n"
        )
    
    report.append("\n### 장기별 상세 정보\n\n")
    
    for rank, (organ_name, stats) in enumerate(sorted_organs, 1):
        report.append(f"#### {rank}. {organ_name.upper()}\n\n")
        report.append(f"**설명**: {stats['description']}\n\n")
        report.append(f"- **총 FP 픽셀**: {stats['total_fp_pixels']:,}개\n")
        report.append(f"- **전체 FP 대비 비율**: {stats['percentage_of_total_fp']:.2f}%\n")
        report.append(f"- **영향받은 케이스**: {stats['cases_affected']}개 ({stats['percentage_of_cases']:.1f}%)\n")
        report.append(f"- **케이스당 평균 FP**: {stats['avg_fp_per_case']:.0f}픽셀\n\n")
    
    # 심각한 오인 케이스
    report.append("\n## 🔴 심각한 오인 케이스 (Top 10)\n\n")
    
    serious_cases = []
    for case_id, result in all_results.items():
        if result['fp_count'] > 0 and result['confused_organs']:
            # 가장 큰 오인 장기 찾기
            max_organ = max(result['confused_organs'].items(), 
                          key=lambda x: x[1]['fp_pixels'])
            serious_cases.append({
                'case_id': case_id,
                'dice': result['dice'],
                'total_fp': result['fp_count'],
                'main_organ': max_organ[0],
                'main_organ_fp': max_organ[1]['fp_pixels'],
                'main_organ_pct': max_organ[1]['percentage']
            })
    
    serious_cases.sort(key=lambda x: x['total_fp'], reverse=True)
    
    report.append("| 케이스 ID | Dice | 총 FP | 주요 오인 장기 | 해당 장기 FP | 비율 |\n")
    report.append("|-----------|------|-------|----------------|--------------|------|\n")
    
    for case in serious_cases[:10]:
        report.append(
            f"| {case['case_id']} | "
            f"{case['dice']:.3f} | "
            f"{case['total_fp']:,} | "
            f"**{case['main_organ'].upper()}** | "
            f"{case['main_organ_fp']:,} | "
            f"{case['main_organ_pct']:.1f}% |\n"
        )
    
    # 핵심 발견사항
    report.append("\n## 💡 핵심 발견사항\n\n")
    
    if sorted_organs:
        top_organ = sorted_organs[0]
        report.append(f"### 1. 가장 많이 혼동하는 장기: **{top_organ[0].upper()}**\n\n")
        report.append(f"- 전체 FP의 **{top_organ[1]['percentage_of_total_fp']:.1f}%**를 차지\n")
        report.append(f"- 전체 케이스 중 **{top_organ[1]['percentage_of_cases']:.1f}%**에서 발생\n")
        report.append(f"- {top_organ[1]['description']}\n\n")
    
    if len(sorted_organs) >= 2:
        second_organ = sorted_organs[1]
        report.append(f"### 2. 두 번째로 많이 혼동하는 장기: **{second_organ[0].upper()}**\n\n")
        report.append(f"- 전체 FP의 **{second_organ[1]['percentage_of_total_fp']:.1f}%**를 차지\n")
        report.append(f"- {second_organ[1]['description']}\n\n")
    
    # 개선 권장사항
    report.append("\n## 🎯 개선 권장사항\n\n")
    
    if sorted_organs:
        top_confused = [organ[0] for organ in sorted_organs[:3]]
        report.append("### 1. Text Prompt 개선\n\n")
        report.append(f"현재 주요 혼동 장기: {', '.join([o.upper() for o in top_confused])}\n\n")
        report.append("**권장 Negative Prompts**:\n")
        for organ in top_confused:
            report.append(f"- `not {organ}`\n")
        report.append("\n")
        
        report.append("**권장 개선된 Prompt**:\n")
        report.append('```\n')
        report.append(f"pancreas only, exclude {' and '.join(top_confused)}\n")
        report.append('```\n\n')
        
        report.append("### 2. 후처리 개선\n\n")
        report.append("- **크기 제한**: 예측 영역이 췌장 전형적 크기(이미지의 0.5%~3%)를 초과하면 재검토\n")
        report.append("- **위치 제한**: 복부 중앙 영역(normalized y: 0.35~0.55, x: 0.30~0.60) 밖의 영역 제거\n")
        report.append("- **형태 제한**: 췌장의 전형적인 elongated 형태와 맞지 않는 영역 필터링\n\n")
    
    report.append("### 3. 앙상블 및 다단계 접근\n\n")
    report.append("- **1단계**: 큰 복부 장기들(간, 비장, 위) 먼저 제외\n")
    report.append("- **2단계**: 남은 영역에서 췌장 영역 탐지\n")
    report.append("- **3단계**: 최종 형태 및 위치 검증\n\n")
    
    # 파일 저장
    with open(output_dir / 'organ_confusion_report.md', 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print("\n보고서 미리보기:")
    print("".join(report[:30]))


if __name__ == '__main__':
    main()
