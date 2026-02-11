"""
장기별 오인율 시각화
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

def load_results():
    """분석 결과 로드"""
    base_dir = Path('/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/evaluation/organ_confusion_analysis')
    
    with open(base_dir / 'organ_confusion_summary.json', 'r') as f:
        summary = json.load(f)
    
    return summary

def create_visualizations(summary):
    """다양한 시각화 생성"""
    
    output_dir = Path('/home/woojye2020/decs_jupyter_lab/MedCLIP-SAMv2/evaluation/organ_confusion_analysis')
    
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    organ_stats = summary['organ_confusion_stats']
    
    # 데이터 준비
    organs = []
    fp_pixels = []
    percentages = []
    cases_affected = []
    
    for organ, stats in sorted(organ_stats.items(), 
                               key=lambda x: x[1]['total_fp_pixels'], 
                               reverse=True):
        organs.append(organ.upper())
        fp_pixels.append(stats['total_fp_pixels'])
        percentages.append(stats['percentage_of_total_fp'])
        cases_affected.append(stats['cases_affected'])
    
    # 1. 장기별 FP 픽셀 수 막대 그래프
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.3, 0.7, len(organs)))
    
    bars = ax.bar(organs, fp_pixels, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Confused Organ', fontsize=14, fontweight='bold')
    ax.set_ylabel('Total FP Pixels', fontsize=14, fontweight='bold')
    ax.set_title('Pancreas Segmentation: FP Pixels by Confused Organ', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # 값 표시
    for bar, val, pct in zip(bars, fp_pixels, percentages):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'fp_pixels_by_organ.png', dpi=300, bbox_inches='tight')
    print(f"Saved: fp_pixels_by_organ.png")
    plt.close()
    
    # 2. 파이 차트 - 전체 FP 비율
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 작은 비율은 'Other'로 합치기
    threshold = 1.0
    main_organs = []
    main_percentages = []
    other_sum = 0
    
    for organ, pct in zip(organs, percentages):
        if pct >= threshold:
            main_organs.append(organ)
            main_percentages.append(pct)
        else:
            other_sum += pct
    
    if other_sum > 0:
        main_organs.append('OTHER')
        main_percentages.append(other_sum)
    
    colors_pie = plt.cm.Set3(np.linspace(0, 1, len(main_organs)))
    
    wedges, texts, autotexts = ax.pie(main_percentages, 
                                       labels=main_organs,
                                       autopct='%1.1f%%',
                                       startangle=90,
                                       colors=colors_pie,
                                       textprops={'fontsize': 11, 'fontweight': 'bold'})
    
    ax.set_title('Distribution of FP Pixels Across Confused Organs', 
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'fp_distribution_pie.png', dpi=300, bbox_inches='tight')
    print(f"Saved: fp_distribution_pie.png")
    plt.close()
    
    # 3. 영향받은 케이스 수 비교
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(organs, cases_affected, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Confused Organ', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Cases Affected', fontsize=14, fontweight='bold')
    ax.set_title('Number of Cases Where Each Organ Was Confused with Pancreas', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # 값 표시
    total_cases = summary['total_cases']
    for bar, val in zip(bars, cases_affected):
        height = bar.get_height()
        pct = val / total_cases * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'cases_affected_by_organ.png', dpi=300, bbox_inches='tight')
    print(f"Saved: cases_affected_by_organ.png")
    plt.close()
    
    # 4. 케이스당 평균 FP 픽셀
    avg_fp_per_case = [organ_stats[organ.lower()]['avg_fp_per_case'] 
                       for organ in organs]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars = ax.bar(organs, avg_fp_per_case, color=colors, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Confused Organ', fontsize=14, fontweight='bold')
    ax.set_ylabel('Average FP Pixels per Case', fontsize=14, fontweight='bold')
    ax.set_title('Average FP Pixels per Case for Each Confused Organ', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # 값 표시
    for bar, val in zip(bars, avg_fp_per_case):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'avg_fp_per_case.png', dpi=300, bbox_inches='tight')
    print(f"Saved: avg_fp_per_case.png")
    plt.close()
    
    # 5. 통합 대시보드
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # 5-1. 총 FP 픽셀
    ax1 = fig.add_subplot(gs[0, 0])
    bars1 = ax1.bar(organs, fp_pixels, color=colors, edgecolor='black')
    ax1.set_ylabel('Total FP Pixels', fontsize=11, fontweight='bold')
    ax1.set_title('Total FP Pixels by Organ', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45, labelsize=9)
    ax1.grid(axis='y', alpha=0.3)
    
    # 5-2. 영향받은 케이스 비율
    ax2 = fig.add_subplot(gs[0, 1])
    case_percentages = [c / total_cases * 100 for c in cases_affected]
    bars2 = ax2.bar(organs, case_percentages, color=colors, edgecolor='black')
    ax2.set_ylabel('Percentage of Cases (%)', fontsize=11, fontweight='bold')
    ax2.set_title('Percentage of Cases Affected', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45, labelsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # 5-3. FP 비율 파이 차트
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.pie(main_percentages, labels=main_organs, autopct='%1.1f%%',
            startangle=90, colors=colors_pie, textprops={'fontsize': 9})
    ax3.set_title('FP Distribution', fontsize=12, fontweight='bold')
    
    # 5-4. 요약 테이블
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('tight')
    ax4.axis('off')
    
    table_data = [['Organ', 'FP Pixels', '% of FP', 'Cases']]
    for organ, fp, pct, cases in zip(organs[:5], fp_pixels[:5], percentages[:5], cases_affected[:5]):
        table_data.append([organ, f'{fp:,}', f'{pct:.1f}%', f'{cases}'])
    
    table = ax4.table(cellText=table_data, cellLoc='left', loc='center',
                     colWidths=[0.3, 0.25, 0.2, 0.15])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # 헤더 스타일
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Top 5 Confused Organs Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.suptitle('Pancreas Segmentation: Organ Confusion Analysis Dashboard', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(output_dir / 'confusion_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"Saved: confusion_dashboard.png")
    plt.close()
    
    print("\nAll visualizations created successfully!")

def main():
    print("Loading analysis results...")
    summary = load_results()
    
    print("\nCreating visualizations...")
    create_visualizations(summary)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total cases analyzed: {summary['total_cases']}")
    print(f"Total FP pixels: {summary['total_fp_pixels']:,}")
    print("\nTop 3 Confused Organs:")
    
    organ_stats = summary['organ_confusion_stats']
    sorted_organs = sorted(organ_stats.items(), 
                          key=lambda x: x[1]['total_fp_pixels'], 
                          reverse=True)[:3]
    
    for i, (organ, stats) in enumerate(sorted_organs, 1):
        print(f"  {i}. {organ.upper()}: {stats['percentage_of_total_fp']:.1f}% of total FP")
        print(f"     - {stats['cases_affected']}/{summary['total_cases']} cases affected")

if __name__ == '__main__':
    main()
