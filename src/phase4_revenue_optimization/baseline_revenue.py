"""
Phase 4.1: Current State Revenue Analysis
Calculates baseline Revenue Per Application (RPA) metrics to establish
pre-optimization revenue performance across lenders and customer segments
"""

import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from phase1_eda.data_loader import load_and_validate

# Setup paths
REPORTS_DIR = Path(__file__).parent.parent.parent / 'reports' / 'phase4_revenue_optimization'
TABLES_DIR = REPORTS_DIR / 'tables'
FIGURES_DIR = REPORTS_DIR / 'figures'

# Create directories
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def prepare_features(df):
    """Prepare engineered features for segmentation"""
    df = df.copy()

    # Income quartiles
    df['Income_Quartile'] = pd.qcut(df['Monthly_Gross_Income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    # Loan brackets
    df['Loan_Bracket'] = pd.cut(
        df['Loan_Amount'],
        bins=[0, 30000, 60000, float('inf')],
        labels=['Small', 'Medium', 'Large']
    )

    return df

def calculate_overall_rpa(df):
    """
    Calculate overall Revenue Per Application (RPA)
    """
    print("\n" + "="*80)
    print("OVERALL REVENUE PER APPLICATION (RPA)")
    print("="*80)

    total_bounty = df['bounty'].sum()
    total_applications = len(df)
    overall_rpa = total_bounty / total_applications

    total_approved = df['Approved'].sum()
    approval_rate = total_approved / total_applications

    # Revenue from approved vs denied
    approved_bounty = df[df['Approved'] == 1]['bounty'].sum()
    denied_bounty = df[df['Approved'] == 0]['bounty'].sum()

    metrics = {
        'Total_Bounty': total_bounty,
        'Total_Applications': total_applications,
        'Overall_RPA': overall_rpa,
        'Total_Approved': total_approved,
        'Approval_Rate': approval_rate,
        'Approved_Bounty': approved_bounty,
        'Denied_Bounty': denied_bounty,
        'Avg_Bounty_Per_App': df['bounty'].mean(),
        'Bounty_Std': df['bounty'].std()
    }

    print(f"\nTotal Applications: {total_applications:,}")
    print(f"Total Bounty: ${total_bounty:,.2f}")
    print(f"Overall RPA: ${overall_rpa:.2f}")
    print(f"Approval Rate: {approval_rate:.2%}")
    print(f"\nBounty by Approval Status:")
    print(f"  Approved: ${approved_bounty:,.2f} ({approved_bounty/total_bounty:.1%})")
    print(f"  Denied: ${denied_bounty:,.2f} ({denied_bounty/total_bounty:.1%})")

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(TABLES_DIR / 'overall_rpa.csv', index=False)
    print(f"\n✓ Overall RPA saved to {TABLES_DIR / 'overall_rpa.csv'}")

    return metrics

def calculate_rpa_by_lender(df):
    """
    Calculate RPA by lender (A, B, C)
    """
    print("\n" + "="*80)
    print("RPA BY LENDER")
    print("="*80)

    lender_metrics = []

    for lender in ['A', 'B', 'C']:
        lender_df = df[df['Lender'] == lender]

        total_apps = len(lender_df)
        total_bounty = lender_df['bounty'].sum()
        rpa = total_bounty / total_apps

        approved = lender_df['Approved'].sum()
        approval_rate = approved / total_apps

        approved_bounty = lender_df[lender_df['Approved'] == 1]['bounty'].sum()
        denied_bounty = lender_df[lender_df['Approved'] == 0]['bounty'].sum()

        avg_bounty = lender_df['bounty'].mean()
        bounty_std = lender_df['bounty'].std()

        lender_metrics.append({
            'Lender': lender,
            'Total_Applications': total_apps,
            'Total_Bounty': total_bounty,
            'RPA': rpa,
            'Approval_Rate': approval_rate,
            'Approved_Bounty': approved_bounty,
            'Denied_Bounty': denied_bounty,
            'Avg_Bounty': avg_bounty,
            'Bounty_Std': bounty_std,
            'Pct_of_Total_Revenue': total_bounty / df['bounty'].sum() * 100
        })

        print(f"\nLender {lender}:")
        print(f"  Applications: {total_apps:,} ({total_apps/len(df):.1%})")
        print(f"  Total Bounty: ${total_bounty:,.2f}")
        print(f"  RPA: ${rpa:.2f}")
        print(f"  Approval Rate: {approval_rate:.2%}")
        print(f"  Avg Bounty: ${avg_bounty:.2f}")

    lender_df = pd.DataFrame(lender_metrics)
    lender_df.to_csv(TABLES_DIR / 'rpa_by_lender.csv', index=False)
    print(f"\n✓ RPA by lender saved to {TABLES_DIR / 'rpa_by_lender.csv'}")

    return lender_df

def calculate_rpa_by_fico_segment(df):
    """
    Calculate RPA by FICO score groups
    """
    print("\n" + "="*80)
    print("RPA BY FICO SEGMENT")
    print("="*80)

    fico_metrics = []

    for fico_group in df['Fico_Score_group'].unique():
        if pd.isna(fico_group):
            continue

        segment_df = df[df['Fico_Score_group'] == fico_group]

        total_apps = len(segment_df)
        total_bounty = segment_df['bounty'].sum()
        rpa = total_bounty / total_apps

        approved = segment_df['Approved'].sum()
        approval_rate = approved / total_apps

        fico_metrics.append({
            'FICO_Group': fico_group,
            'Total_Applications': total_apps,
            'Total_Bounty': total_bounty,
            'RPA': rpa,
            'Approval_Rate': approval_rate,
            'Avg_Bounty': segment_df['bounty'].mean(),
            'Pct_of_Total_Apps': total_apps / len(df) * 100
        })

        print(f"\n{fico_group}:")
        print(f"  Applications: {total_apps:,} ({total_apps/len(df):.1%})")
        print(f"  RPA: ${rpa:.2f}")
        print(f"  Approval Rate: {approval_rate:.2%}")

    fico_df = pd.DataFrame(fico_metrics)
    # Sort by FICO quality
    fico_order = ['poor', 'fair', 'good', 'very_good', 'exceptional']
    fico_df['FICO_Group'] = pd.Categorical(fico_df['FICO_Group'], categories=fico_order, ordered=True)
    fico_df = fico_df.sort_values('FICO_Group')

    fico_df.to_csv(TABLES_DIR / 'rpa_by_fico_segment.csv', index=False)
    print(f"\n✓ RPA by FICO segment saved to {TABLES_DIR / 'rpa_by_fico_segment.csv'}")

    return fico_df

def calculate_rpa_by_income_segment(df):
    """
    Calculate RPA by income quartiles
    """
    print("\n" + "="*80)
    print("RPA BY INCOME SEGMENT")
    print("="*80)

    income_metrics = []

    for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
        segment_df = df[df['Income_Quartile'] == quartile]

        total_apps = len(segment_df)
        total_bounty = segment_df['bounty'].sum()
        rpa = total_bounty / total_apps

        approved = segment_df['Approved'].sum()
        approval_rate = approved / total_apps

        income_metrics.append({
            'Income_Quartile': quartile,
            'Total_Applications': total_apps,
            'Total_Bounty': total_bounty,
            'RPA': rpa,
            'Approval_Rate': approval_rate,
            'Avg_Bounty': segment_df['bounty'].mean(),
            'Avg_Income': segment_df['Monthly_Gross_Income'].mean(),
            'Pct_of_Total_Apps': total_apps / len(df) * 100
        })

        print(f"\n{quartile}:")
        print(f"  Applications: {total_apps:,} ({total_apps/len(df):.1%})")
        print(f"  Avg Income: ${segment_df['Monthly_Gross_Income'].mean():,.2f}")
        print(f"  RPA: ${rpa:.2f}")
        print(f"  Approval Rate: {approval_rate:.2%}")

    income_df = pd.DataFrame(income_metrics)
    income_df.to_csv(TABLES_DIR / 'rpa_by_income_segment.csv', index=False)
    print(f"\n✓ RPA by income segment saved to {TABLES_DIR / 'rpa_by_income_segment.csv'}")

    return income_df

def calculate_rpa_by_loan_bracket(df):
    """
    Calculate RPA by loan amount brackets
    """
    print("\n" + "="*80)
    print("RPA BY LOAN BRACKET")
    print("="*80)

    loan_metrics = []

    for bracket in ['Small', 'Medium', 'Large']:
        segment_df = df[df['Loan_Bracket'] == bracket]

        total_apps = len(segment_df)
        total_bounty = segment_df['bounty'].sum()
        rpa = total_bounty / total_apps

        approved = segment_df['Approved'].sum()
        approval_rate = approved / total_apps

        loan_metrics.append({
            'Loan_Bracket': bracket,
            'Total_Applications': total_apps,
            'Total_Bounty': total_bounty,
            'RPA': rpa,
            'Approval_Rate': approval_rate,
            'Avg_Bounty': segment_df['bounty'].mean(),
            'Avg_Loan_Amount': segment_df['Loan_Amount'].mean(),
            'Pct_of_Total_Apps': total_apps / len(df) * 100
        })

        print(f"\n{bracket}:")
        print(f"  Applications: {total_apps:,} ({total_apps/len(df):.1%})")
        print(f"  Avg Loan: ${segment_df['Loan_Amount'].mean():,.2f}")
        print(f"  RPA: ${rpa:.2f}")
        print(f"  Approval Rate: {approval_rate:.2%}")

    loan_df = pd.DataFrame(loan_metrics)
    loan_df.to_csv(TABLES_DIR / 'rpa_by_loan_bracket.csv', index=False)
    print(f"\n✓ RPA by loan bracket saved to {TABLES_DIR / 'rpa_by_loan_bracket.csv'}")

    return loan_df

def analyze_bounty_structure(df):
    """
    Analyze bounty distribution and structure
    """
    print("\n" + "="*80)
    print("BOUNTY STRUCTURE ANALYSIS")
    print("="*80)

    # Check if bounty is fixed or variable
    unique_bounties = df['bounty'].nunique()
    bounty_mean = df['bounty'].mean()
    bounty_std = df['bounty'].std()
    bounty_min = df['bounty'].min()
    bounty_max = df['bounty'].max()

    print(f"\nBounty Statistics:")
    print(f"  Unique Values: {unique_bounties}")
    print(f"  Mean: ${bounty_mean:.2f}")
    print(f"  Std Dev: ${bounty_std:.2f}")
    print(f"  Min: ${bounty_min:.2f}")
    print(f"  Max: ${bounty_max:.2f}")

    # Bounty by lender
    print(f"\nBounty by Lender:")
    for lender in ['A', 'B', 'C']:
        lender_bounty = df[df['Lender'] == lender]['bounty']
        print(f"  Lender {lender}: Mean=${lender_bounty.mean():.2f}, Std=${lender_bounty.std():.2f}")

    # Bounty by approval status
    print(f"\nBounty by Approval Status:")
    for status in [0, 1]:
        status_bounty = df[df['Approved'] == status]['bounty']
        status_label = 'Approved' if status == 1 else 'Denied'
        print(f"  {status_label}: Mean=${status_bounty.mean():.2f}, Std=${status_bounty.std():.2f}")

    bounty_structure = {
        'unique_values': unique_bounties,
        'mean': bounty_mean,
        'std': bounty_std,
        'min': bounty_min,
        'max': bounty_max,
        'is_fixed': unique_bounties == 1,
        'coefficient_of_variation': (bounty_std / bounty_mean) if bounty_mean > 0 else 0
    }

    # Save structure analysis
    structure_df = pd.DataFrame([bounty_structure])
    structure_df.to_csv(TABLES_DIR / 'bounty_structure.csv', index=False)
    print(f"\n✓ Bounty structure saved to {TABLES_DIR / 'bounty_structure.csv'}")

    return bounty_structure

def create_baseline_visualizations(overall_metrics, lender_df, fico_df, income_df, loan_df):
    """
    Create comprehensive baseline revenue visualizations
    """
    print("\n" + "="*80)
    print("CREATING BASELINE REVENUE VISUALIZATIONS")
    print("="*80)

    plt.style.use('default')
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    # 1. Overall RPA breakdown (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    rpa_data = [overall_metrics['Overall_RPA']]
    ax1.bar(['Overall RPA'], rpa_data, color='#3498db', width=0.4)
    ax1.set_ylabel('RPA ($)', fontsize=11)
    ax1.set_title('Overall Revenue Per Application', fontsize=12, fontweight='bold')
    ax1.axhline(y=overall_metrics['Overall_RPA'], color='red', linestyle='--', alpha=0.3)
    for i, v in enumerate(rpa_data):
        ax1.text(i, v + 0.5, f'${v:.2f}', ha='center', va='bottom', fontweight='bold')

    # 2. RPA by Lender (top-middle)
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    ax2.bar(lender_df['Lender'], lender_df['RPA'], color=colors)
    ax2.set_xlabel('Lender', fontsize=11)
    ax2.set_ylabel('RPA ($)', fontsize=11)
    ax2.set_title('RPA by Lender', fontsize=12, fontweight='bold')
    ax2.axhline(y=overall_metrics['Overall_RPA'], color='gray', linestyle='--', alpha=0.5, label='Overall Avg')
    ax2.legend(fontsize=9)
    for i, (lender, rpa) in enumerate(zip(lender_df['Lender'], lender_df['RPA'])):
        ax2.text(i, rpa + 0.5, f'${rpa:.2f}', ha='center', va='bottom', fontweight='bold')

    # 3. Revenue split by lender (top-right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.pie(lender_df['Pct_of_Total_Revenue'], labels=lender_df['Lender'], autopct='%1.1f%%',
            colors=colors, startangle=90)
    ax3.set_title('Revenue Distribution by Lender', fontsize=12, fontweight='bold')

    # 4. RPA by FICO segment (middle-left)
    ax4 = fig.add_subplot(gs[1, 0])
    fico_colors = ['#e74c3c', '#e67e22', '#f39c12', '#2ecc71', '#27ae60']
    ax4.bar(range(len(fico_df)), fico_df['RPA'], color=fico_colors)
    ax4.set_xlabel('FICO Score Group', fontsize=11)
    ax4.set_ylabel('RPA ($)', fontsize=11)
    ax4.set_title('RPA by FICO Segment', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(len(fico_df)))
    ax4.set_xticklabels(fico_df['FICO_Group'], rotation=45, ha='right')
    ax4.axhline(y=overall_metrics['Overall_RPA'], color='gray', linestyle='--', alpha=0.5)

    # 5. RPA by Income Quartile (middle-middle)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.bar(income_df['Income_Quartile'], income_df['RPA'], color='#9b59b6')
    ax5.set_xlabel('Income Quartile', fontsize=11)
    ax5.set_ylabel('RPA ($)', fontsize=11)
    ax5.set_title('RPA by Income Segment', fontsize=12, fontweight='bold')
    ax5.axhline(y=overall_metrics['Overall_RPA'], color='gray', linestyle='--', alpha=0.5)
    for i, (q, rpa) in enumerate(zip(income_df['Income_Quartile'], income_df['RPA'])):
        ax5.text(i, rpa + 0.5, f'${rpa:.2f}', ha='center', va='bottom', fontweight='bold')

    # 6. RPA by Loan Bracket (middle-right)
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.bar(loan_df['Loan_Bracket'], loan_df['RPA'], color='#e67e22')
    ax6.set_xlabel('Loan Amount Bracket', fontsize=11)
    ax6.set_ylabel('RPA ($)', fontsize=11)
    ax6.set_title('RPA by Loan Size', fontsize=12, fontweight='bold')
    ax6.axhline(y=overall_metrics['Overall_RPA'], color='gray', linestyle='--', alpha=0.5)
    for i, (bracket, rpa) in enumerate(zip(loan_df['Loan_Bracket'], loan_df['RPA'])):
        ax6.text(i, rpa + 0.5, f'${rpa:.2f}', ha='center', va='bottom', fontweight='bold')

    # 7. Approval Rate vs RPA by Lender (bottom-left)
    ax7 = fig.add_subplot(gs[2, 0])
    x = lender_df['Approval_Rate'] * 100
    y = lender_df['RPA']
    ax7.scatter(x, y, s=200, c=colors, alpha=0.6, edgecolors='black')
    for i, lender in enumerate(lender_df['Lender']):
        ax7.annotate(f'Lender {lender}', (x.iloc[i], y.iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    ax7.set_xlabel('Approval Rate (%)', fontsize=11)
    ax7.set_ylabel('RPA ($)', fontsize=11)
    ax7.set_title('Approval Rate vs RPA by Lender', fontsize=12, fontweight='bold')
    ax7.grid(alpha=0.3)

    # 8. Application volume by segment (bottom-middle)
    ax8 = fig.add_subplot(gs[2, 1])
    segment_apps = [
        len(fico_df),
        len(income_df),
        len(loan_df),
        len(lender_df)
    ]
    segment_labels = ['FICO\nGroups', 'Income\nQuartiles', 'Loan\nBrackets', 'Lenders']
    ax8.bar(segment_labels, [5, 4, 3, 3], color='#95a5a6')  # Number of segments
    ax8.set_ylabel('Number of Segments', fontsize=11)
    ax8.set_title('Segmentation Coverage', fontsize=12, fontweight='bold')
    ax8.set_ylim(0, 6)

    # 9. Revenue metrics summary (bottom-right)
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    summary_text = f"""
    BASELINE REVENUE METRICS

    Total Applications: {overall_metrics['Total_Applications']:,}
    Total Revenue: ${overall_metrics['Total_Bounty']:,.2f}
    Overall RPA: ${overall_metrics['Overall_RPA']:.2f}

    Approval Rate: {overall_metrics['Approval_Rate']:.1%}

    Revenue from Approved: ${overall_metrics['Approved_Bounty']:,.2f}
    Revenue from Denied: ${overall_metrics['Denied_Bounty']:,.2f}

    Lender RPA Range:
      High: ${lender_df['RPA'].max():.2f} (Lender {lender_df.loc[lender_df['RPA'].idxmax(), 'Lender']})
      Low: ${lender_df['RPA'].min():.2f} (Lender {lender_df.loc[lender_df['RPA'].idxmin(), 'Lender']})
    """
    ax9.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('Phase 4.1: Baseline Revenue Analysis Dashboard', fontsize=14, fontweight='bold', y=0.995)

    viz_path = FIGURES_DIR / 'baseline_revenue_dashboard.png'
    try:
        plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"✓ Baseline dashboard saved to {viz_path}")
    except Exception as e:
        print(f"⚠️  Error saving visualization: {str(e)}")
    finally:
        plt.close()

def main():
    """
    Execute Phase 4.1: Current State Revenue Analysis
    """
    start_time = time.time()

    print("="*80)
    print("PHASE 4.1: CURRENT STATE REVENUE ANALYSIS")
    print("="*80)

    try:
        # 1. Load and prepare data
        df, report = load_and_validate()
        df = prepare_features(df)

        # 2. Calculate overall RPA
        overall_metrics = calculate_overall_rpa(df)

        # 3. Calculate RPA by lender
        lender_df = calculate_rpa_by_lender(df)

        # 4. Calculate RPA by FICO segment
        fico_df = calculate_rpa_by_fico_segment(df)

        # 5. Calculate RPA by income segment
        income_df = calculate_rpa_by_income_segment(df)

        # 6. Calculate RPA by loan bracket
        loan_df = calculate_rpa_by_loan_bracket(df)

        # 7. Analyze bounty structure
        bounty_structure = analyze_bounty_structure(df)

        # 8. Create visualizations
        create_baseline_visualizations(overall_metrics, lender_df, fico_df, income_df, loan_df)

        execution_time = time.time() - start_time

        # Build structured output
        output = {
            "success": True,
            "subphase": "Phase 4.1: Current State Revenue Analysis",
            "execution_time": round(execution_time, 2),
            "summary": {
                "overall_rpa": round(float(overall_metrics['Overall_RPA']), 2),
                "total_revenue": round(float(overall_metrics['Total_Bounty']), 2),
                "total_applications": int(overall_metrics['Total_Applications']),
                "approval_rate": round(float(overall_metrics['Approval_Rate']), 4),
                "lender_rpa_range": {
                    "min": round(float(lender_df['RPA'].min()), 2),
                    "max": round(float(lender_df['RPA'].max()), 2)
                },
                "bounty_is_fixed": bool(bounty_structure['is_fixed'])
            },
            "outputs": {
                "tables": [
                    "overall_rpa.csv",
                    "rpa_by_lender.csv",
                    "rpa_by_fico_segment.csv",
                    "rpa_by_income_segment.csv",
                    "rpa_by_loan_bracket.csv",
                    "bounty_structure.csv"
                ],
                "figures": [
                    "baseline_revenue_dashboard.png"
                ]
            },
            "insights": [
                f"Overall RPA: ${overall_metrics['Overall_RPA']:.2f} per application",
                f"Total baseline revenue: ${overall_metrics['Total_Bounty']:,.2f}",
                f"Lender RPA varies from ${lender_df['RPA'].min():.2f} to ${lender_df['RPA'].max():.2f}",
                f"Approval rate: {overall_metrics['Approval_Rate']:.1%}",
                f"Bounty structure: {'Fixed' if bounty_structure['is_fixed'] else 'Variable'}"
            ]
        }

        # Output JSON for dashboard consumption
        print("\n" + "="*80)
        print("__JSON_OUTPUT__")
        print(json.dumps(output, indent=2))
        print("__JSON_OUTPUT_END__")

        print("="*80)
        print(f"✓ Phase 4.1 complete in {execution_time:.2f}s")
        print("="*80)

        return output

    except Exception as e:
        error_output = {
            "success": False,
            "subphase": "Phase 4.1: Current State Revenue Analysis",
            "error": str(e),
            "error_type": type(e).__name__
        }

        print("\n__JSON_OUTPUT__")
        print(json.dumps(error_output, indent=2))
        print("__JSON_OUTPUT_END__")

        raise

if __name__ == "__main__":
    main()
