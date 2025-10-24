"""
Phase 3.1: Lender Approval Profiling
Characterize each lender's approval patterns and identify differences
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from phase1_eda.data_loader import load_and_validate

def calculate_approval_rates_by_segment(df, lender):
    """Calculate approval rates by various segments for a specific lender"""
    lender_df = df[df['Lender'] == lender].copy()

    segments = {}

    # 1. FICO score groups
    segments['fico_groups'] = lender_df.groupby('Fico_Score_group')['Approved'].agg([
        ('Total', 'count'),
        ('Approved', 'sum'),
        ('Approval_Rate', 'mean')
    ]).reset_index()

    # 2. Income quartiles
    lender_df['Income_Quartile'] = pd.qcut(lender_df['Monthly_Gross_Income'], q=4,
                                            labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    segments['income_quartiles'] = lender_df.groupby('Income_Quartile')['Approved'].agg([
        ('Total', 'count'),
        ('Approved', 'sum'),
        ('Approval_Rate', 'mean')
    ]).reset_index()

    # 3. Loan amount brackets
    lender_df['Loan_Bracket'] = pd.cut(lender_df['Loan_Amount'], bins=5,
                                        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    segments['loan_brackets'] = lender_df.groupby('Loan_Bracket')['Approved'].agg([
        ('Total', 'count'),
        ('Approved', 'sum'),
        ('Approval_Rate', 'mean')
    ]).reset_index()

    # 4. Employment status
    segments['employment_status'] = lender_df.groupby('Employment_Status')['Approved'].agg([
        ('Total', 'count'),
        ('Approved', 'sum'),
        ('Approval_Rate', 'mean')
    ]).reset_index()

    # 5. Loan reason
    segments['loan_reason'] = lender_df.groupby('Reason')['Approved'].agg([
        ('Total', 'count'),
        ('Approved', 'sum'),
        ('Approval_Rate', 'mean')
    ]).reset_index()

    # 6. Bankruptcy history
    segments['bankruptcy'] = lender_df.groupby('Ever_Bankrupt_or_Foreclose')['Approved'].agg([
        ('Total', 'count'),
        ('Approved', 'sum'),
        ('Approval_Rate', 'mean')
    ]).reset_index()

    return segments


def calculate_statistical_comparisons(df, lender):
    """Calculate statistical comparisons between approved and denied applications"""
    lender_df = df[df['Lender'] == lender].copy()

    approved = lender_df[lender_df['Approved'] == 1]
    denied = lender_df[lender_df['Approved'] == 0]

    # Calculate DTI
    lender_df['DTI'] = (lender_df['Monthly_Housing_Payment'] / lender_df['Monthly_Gross_Income']).replace([np.inf, -np.inf], 0).fillna(0)
    approved_dti = approved['Monthly_Housing_Payment'] / approved['Monthly_Gross_Income']
    denied_dti = denied['Monthly_Housing_Payment'] / denied['Monthly_Gross_Income']
    approved_dti = approved_dti.replace([np.inf, -np.inf], 0).fillna(0)
    denied_dti = denied_dti.replace([np.inf, -np.inf], 0).fillna(0)

    comparisons = {
        'FICO_score': {
            'approved_mean': approved['FICO_score'].mean(),
            'denied_mean': denied['FICO_score'].mean(),
            'difference': approved['FICO_score'].mean() - denied['FICO_score'].mean(),
            't_statistic': stats.ttest_ind(approved['FICO_score'], denied['FICO_score'])[0],
            'p_value': stats.ttest_ind(approved['FICO_score'], denied['FICO_score'])[1]
        },
        'Monthly_Gross_Income': {
            'approved_mean': approved['Monthly_Gross_Income'].mean(),
            'denied_mean': denied['Monthly_Gross_Income'].mean(),
            'difference': approved['Monthly_Gross_Income'].mean() - denied['Monthly_Gross_Income'].mean(),
            't_statistic': stats.ttest_ind(approved['Monthly_Gross_Income'], denied['Monthly_Gross_Income'])[0],
            'p_value': stats.ttest_ind(approved['Monthly_Gross_Income'], denied['Monthly_Gross_Income'])[1]
        },
        'Loan_Amount': {
            'approved_mean': approved['Loan_Amount'].mean(),
            'denied_mean': denied['Loan_Amount'].mean(),
            'difference': approved['Loan_Amount'].mean() - denied['Loan_Amount'].mean(),
            't_statistic': stats.ttest_ind(approved['Loan_Amount'], denied['Loan_Amount'])[0],
            'p_value': stats.ttest_ind(approved['Loan_Amount'], denied['Loan_Amount'])[1]
        },
        'DTI': {
            'approved_mean': approved_dti.mean(),
            'denied_mean': denied_dti.mean(),
            'difference': approved_dti.mean() - denied_dti.mean(),
            't_statistic': stats.ttest_ind(approved_dti, denied_dti)[0],
            'p_value': stats.ttest_ind(approved_dti, denied_dti)[1]
        }
    }

    return comparisons


def detect_approval_thresholds(df, lender):
    """Detect approval thresholds for a specific lender"""
    lender_df = df[df['Lender'] == lender].copy()
    approved = lender_df[lender_df['Approved'] == 1]

    # Calculate LTI
    approved['LTI'] = (approved['Loan_Amount'] / approved['Monthly_Gross_Income']).replace([np.inf, -np.inf], 0).fillna(0)

    thresholds = {
        'min_fico': approved['FICO_score'].quantile(0.05),
        'min_income': approved['Monthly_Gross_Income'].quantile(0.05),
        'max_lti': approved['LTI'].quantile(0.95),
        'bankruptcy_approval_rate': lender_df[lender_df['Ever_Bankrupt_or_Foreclose'] == 1]['Approved'].mean()
    }

    return thresholds


def create_segment_heatmap(df, lender):
    """Create approval rate heatmap for lender segments"""
    lender_df = df[df['Lender'] == lender].copy()

    # Create pivot table for FICO groups vs Income quartiles
    lender_df['Income_Quartile'] = pd.qcut(lender_df['Monthly_Gross_Income'], q=4,
                                            labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')

    pivot = lender_df.groupby(['Fico_Score_group', 'Income_Quartile'])['Approved'].mean().unstack()

    # Reorder FICO groups
    fico_order = ['poor', 'fair', 'good', 'very_good', 'exceptional']
    pivot = pivot.reindex([g for g in fico_order if g in pivot.index])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.2%', cmap='RdYlGn', center=0.15,
                cbar_kws={'label': 'Approval Rate'}, ax=ax)
    ax.set_title(f'Lender {lender}: Approval Rate by FICO Group and Income Quartile',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Income Quartile', fontsize=11)
    ax.set_ylabel('FICO Score Group', fontsize=11)

    plt.tight_layout()
    plt.savefig(f'reports/phase3_lender_analysis/figures/lender_{lender}_heatmap.png',
                dpi=300, bbox_inches='tight')
    plt.close()


def create_lender_comparison_charts(df, lender_stats):
    """Create comparison charts across all lenders"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    lenders = ['A', 'B', 'C']

    # 1. Overall approval rates
    ax1 = axes[0, 0]
    approval_rates = [lender_stats[l]['overall_approval_rate'] for l in lenders]
    bars = ax1.bar(lenders, approval_rates, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
    ax1.set_ylabel('Approval Rate', fontsize=10)
    ax1.set_title('Overall Approval Rates by Lender', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, max(approval_rates) * 1.2])
    ax1.grid(True, alpha=0.3, axis='y')

    for i, (bar, rate) in enumerate(zip(bars, approval_rates)):
        ax1.text(i, rate + 0.005, f'{rate:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. Mean FICO scores (Approved vs Denied)
    ax2 = axes[0, 1]
    approved_fico = [lender_stats[l]['comparisons']['FICO_score']['approved_mean'] for l in lenders]
    denied_fico = [lender_stats[l]['comparisons']['FICO_score']['denied_mean'] for l in lenders]

    x = np.arange(len(lenders))
    width = 0.35
    ax2.bar(x - width/2, approved_fico, width, label='Approved', alpha=0.8, color='green')
    ax2.bar(x + width/2, denied_fico, width, label='Denied', alpha=0.8, color='red')
    ax2.set_ylabel('Mean FICO Score', fontsize=10)
    ax2.set_title('Mean FICO Score: Approved vs Denied', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(lenders)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Approval thresholds (Min FICO)
    ax3 = axes[1, 0]
    min_fico_thresholds = [lender_stats[l]['thresholds']['min_fico'] for l in lenders]
    bars = ax3.bar(lenders, min_fico_thresholds, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
    ax3.set_ylabel('FICO Score (5th Percentile)', fontsize=10)
    ax3.set_title('Minimum FICO Score Threshold (Approvals)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    for i, (bar, thresh) in enumerate(zip(bars, min_fico_thresholds)):
        ax3.text(i, thresh + 5, f'{thresh:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 4. Bankruptcy tolerance
    ax4 = axes[1, 1]
    bankruptcy_rates = [lender_stats[l]['thresholds']['bankruptcy_approval_rate'] for l in lenders]
    bars = ax4.bar(lenders, bankruptcy_rates, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
    ax4.set_ylabel('Approval Rate', fontsize=10)
    ax4.set_title('Approval Rate for Bankruptcy History', fontsize=12, fontweight='bold')
    ax4.set_ylim([0, max(bankruptcy_rates) * 1.3 if max(bankruptcy_rates) > 0 else 0.1])
    ax4.grid(True, alpha=0.3, axis='y')

    for i, (bar, rate) in enumerate(zip(bars, bankruptcy_rates)):
        ax4.text(i, rate + 0.005, f'{rate:.2%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('reports/phase3_lender_analysis/figures/lender_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()


def run_lender_profiling():
    """Main execution function for Phase 3.1"""
    start_time = time.time()

    print("\n" + "="*70)
    print("Phase 3.1: Lender Approval Profiling")
    print("="*70)

    # Create output directories
    Path('reports/phase3_lender_analysis/tables').mkdir(parents=True, exist_ok=True)
    Path('reports/phase3_lender_analysis/figures').mkdir(parents=True, exist_ok=True)
    print("✓ Output directories created")

    # 1. Load data
    print("\n1. Loading data...")
    df, report = load_and_validate()

    # 2. Overall lender statistics
    print("\n2. Calculating overall lender statistics...")
    lender_stats = {}
    lenders = ['A', 'B', 'C']

    overall_rates = []
    for lender in lenders:
        lender_df = df[df['Lender'] == lender]
        approval_rate = lender_df['Approved'].mean()
        overall_rates.append({
            'Lender': lender,
            'Total_Applications': len(lender_df),
            'Approved': lender_df['Approved'].sum(),
            'Approval_Rate': approval_rate
        })
        print(f"  Lender {lender}: {approval_rate:.2%} approval rate ({lender_df['Approved'].sum():,}/{len(lender_df):,})")

    overall_df = pd.DataFrame(overall_rates)
    overall_df.to_csv('reports/phase3_lender_analysis/tables/lender_approval_rates.csv', index=False)

    # 3. Analyze each lender
    for lender in lenders:
        print(f"\n{'='*60}")
        print(f"ANALYZING LENDER {lender}")
        print(f"{'='*60}")

        # Segment analysis
        print(f"\n3.{lenders.index(lender)+1}. Calculating segment approval rates for Lender {lender}...")
        segments = calculate_approval_rates_by_segment(df, lender)

        lender_stats[lender] = {
            'overall_approval_rate': df[df['Lender'] == lender]['Approved'].mean(),
            'segments': segments
        }

        # Save segment tables
        for segment_name, segment_df in segments.items():
            segment_df.to_csv(f'reports/phase3_lender_analysis/tables/lender_{lender}_{segment_name}.csv', index=False)
            print(f"  ✓ Saved {segment_name} analysis")

        # Statistical comparisons
        print(f"\n  Calculating statistical comparisons for Lender {lender}...")
        comparisons = calculate_statistical_comparisons(df, lender)
        lender_stats[lender]['comparisons'] = comparisons

        # Save comparisons
        comp_df = pd.DataFrame({
            'Metric': list(comparisons.keys()),
            'Approved_Mean': [comparisons[k]['approved_mean'] for k in comparisons.keys()],
            'Denied_Mean': [comparisons[k]['denied_mean'] for k in comparisons.keys()],
            'Difference': [comparisons[k]['difference'] for k in comparisons.keys()],
            'P_Value': [comparisons[k]['p_value'] for k in comparisons.keys()],
            'Significant': [comparisons[k]['p_value'] < 0.01 for k in comparisons.keys()]
        })
        comp_df.to_csv(f'reports/phase3_lender_analysis/tables/lender_{lender}_comparisons.csv', index=False)
        print(f"  ✓ Saved statistical comparisons")

        # Threshold detection
        print(f"\n  Detecting approval thresholds for Lender {lender}...")
        thresholds = detect_approval_thresholds(df, lender)
        lender_stats[lender]['thresholds'] = thresholds

        thresh_df = pd.DataFrame([thresholds])
        thresh_df.insert(0, 'Lender', lender)
        thresh_df.to_csv(f'reports/phase3_lender_analysis/tables/lender_{lender}_thresholds.csv', index=False)
        print(f"  ✓ Saved approval thresholds")
        print(f"    Min FICO: {thresholds['min_fico']:.0f}")
        print(f"    Min Income: ${thresholds['min_income']:,.0f}")
        print(f"    Max LTI: {thresholds['max_lti']:.2f}")
        print(f"    Bankruptcy tolerance: {thresholds['bankruptcy_approval_rate']:.2%}")

        # Create heatmap
        print(f"\n  Creating segment heatmap for Lender {lender}...")
        create_segment_heatmap(df, lender)
        print(f"  ✓ Heatmap saved")

    # 4. Cross-lender comparison
    print("\n4. Creating cross-lender comparison charts...")
    create_lender_comparison_charts(df, lender_stats)
    print("✓ Comparison charts saved")

    # 5. Statistical test for lender differences
    print("\n5. Testing for significant differences between lenders...")
    lender_groups = [df[df['Lender'] == l]['Approved'] for l in lenders]
    f_stat, p_value = stats.f_oneway(*lender_groups)
    significant = p_value < 0.01

    print(f"  F-statistic: {f_stat:.4f}")
    print(f"  P-value: {p_value:.6f}")
    print(f"  Significant difference (p < 0.01): {'Yes ✓' if significant else 'No ✗'}")

    execution_time = time.time() - start_time

    # Build structured output
    output = {
        "success": True,
        "subphase": "Phase 3.1: Lender Approval Profiling",
        "summary": {
            "lenders_analyzed": len(lenders),
            "total_applications": len(df),
            "lender_approval_rates": {
                lender: float(lender_stats[lender]['overall_approval_rate'])
                for lender in lenders
            },
            "significant_differences": bool(significant),
            "p_value": float(p_value)
        },
        "insights": [
            f"Analyzed {len(lenders)} lenders with {len(df):,} total applications",
            f"Lender A: {lender_stats['A']['overall_approval_rate']:.2%} approval rate",
            f"Lender B: {lender_stats['B']['overall_approval_rate']:.2%} approval rate",
            f"Lender C: {lender_stats['C']['overall_approval_rate']:.2%} approval rate",
            f"Statistical test: {'Lenders differ significantly (p<0.01)' if significant else 'No significant difference'}",
            f"Min FICO thresholds - A: {lender_stats['A']['thresholds']['min_fico']:.0f}, B: {lender_stats['B']['thresholds']['min_fico']:.0f}, C: {lender_stats['C']['thresholds']['min_fico']:.0f}",
            "Generated 3 lender heatmaps and comprehensive comparison charts"
        ],
        "outputs": {
            "tables": [
                "lender_approval_rates.csv",
                "lender_A_fico_groups.csv", "lender_A_comparisons.csv", "lender_A_thresholds.csv",
                "lender_B_fico_groups.csv", "lender_B_comparisons.csv", "lender_B_thresholds.csv",
                "lender_C_fico_groups.csv", "lender_C_comparisons.csv", "lender_C_thresholds.csv"
            ],
            "figures": [
                "lender_A_heatmap.png",
                "lender_B_heatmap.png",
                "lender_C_heatmap.png",
                "lender_comparison.png"
            ]
        },
        "execution_time": execution_time
    }

    print("\n" + "="*70)
    print("__JSON_OUTPUT__")
    print(json.dumps(output, indent=2))
    print("="*70)

    return output


if __name__ == "__main__":
    run_lender_profiling()
