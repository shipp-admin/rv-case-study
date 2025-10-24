"""
Phase 1.1: Univariate Analysis Module

Analyzes individual variables and their relationship with loan approval.
Generates approval rate tables and distribution plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


def analyze_categorical_variable(
    df: pd.DataFrame,
    variable: str,
    target: str = 'Approved',
    output_dir: str = 'reports/phase1_eda'
) -> pd.DataFrame:
    """Analyze categorical variable relationship with approval.

    Args:
        df: DataFrame with data
        variable: Name of categorical variable to analyze
        target: Target variable (default: 'Approved')
        output_dir: Directory to save outputs

    Returns:
        DataFrame with approval rates by category
    """
    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/figures").mkdir(exist_ok=True)
    Path(f"{output_dir}/tables").mkdir(exist_ok=True)

    # Calculate approval rate by category
    approval_by_cat = df.groupby(variable)[target].agg([
        ('Total_Applications', 'count'),
        ('Total_Approvals', 'sum'),
        ('Approval_Rate', 'mean')
    ]).reset_index()

    # Sort by approval rate
    approval_by_cat = approval_by_cat.sort_values('Approval_Rate', ascending=False)

    # Save table
    table_path = f"{output_dir}/tables/approval_by_{variable.lower()}.csv"
    approval_by_cat.to_csv(table_path, index=False)
    print(f"✅ Saved: {table_path}")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Approval Rate by Category
    ax1.barh(approval_by_cat[variable], approval_by_cat['Approval_Rate'], color='steelblue')
    ax1.set_xlabel('Approval Rate')
    ax1.set_title(f'Approval Rate by {variable}')
    ax1.set_xlim(0, max(approval_by_cat['Approval_Rate']) * 1.1)

    # Add value labels
    for i, v in enumerate(approval_by_cat['Approval_Rate']):
        ax1.text(v + 0.005, i, f'{v:.1%}', va='center')

    # Plot 2: Application Volume by Category
    ax2.barh(approval_by_cat[variable], approval_by_cat['Total_Applications'], color='coral')
    ax2.set_xlabel('Number of Applications')
    ax2.set_title(f'Application Volume by {variable}')

    plt.tight_layout()
    fig_path = f"{output_dir}/figures/approval_by_{variable.lower()}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fig_path}")

    return approval_by_cat


def analyze_numerical_variable(
    df: pd.DataFrame,
    variable: str,
    target: str = 'Approved',
    output_dir: str = 'reports/phase1_eda'
) -> Dict:
    """Analyze numerical variable relationship with approval.

    Args:
        df: DataFrame with data
        variable: Name of numerical variable to analyze
        target: Target variable (default: 'Approved')
        output_dir: Directory to save outputs

    Returns:
        Dictionary with analysis results
    """
    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(f"{output_dir}/figures").mkdir(exist_ok=True)
    Path(f"{output_dir}/tables").mkdir(exist_ok=True)

    # Split by approval status
    approved = df[df[target] == 1][variable]
    rejected = df[df[target] == 0][variable]

    # Calculate statistics
    stats = {
        'variable': variable,
        'approved_mean': approved.mean(),
        'approved_median': approved.median(),
        'approved_std': approved.std(),
        'rejected_mean': rejected.mean(),
        'rejected_median': rejected.median(),
        'rejected_std': rejected.std(),
        'mean_difference': approved.mean() - rejected.mean(),
        'median_difference': approved.median() - rejected.median()
    }

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Distribution comparison
    ax1.hist(approved, bins=50, alpha=0.6, label='Approved', color='green', density=True)
    ax1.hist(rejected, bins=50, alpha=0.6, label='Rejected', color='red', density=True)
    ax1.set_xlabel(variable)
    ax1.set_ylabel('Density')
    ax1.set_title(f'{variable} Distribution by Approval Status')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Box plot comparison
    box_data = [rejected, approved]
    ax2.boxplot(box_data, labels=['Rejected', 'Approved'], patch_artist=True,
                boxprops=dict(facecolor='lightblue'))
    ax2.set_ylabel(variable)
    ax2.set_title(f'{variable} by Approval Status')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = f"{output_dir}/figures/{variable.lower()}_by_approval.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fig_path}")

    return stats


def analyze_fico_bins(
    df: pd.DataFrame,
    target: str = 'Approved',
    output_dir: str = 'reports/phase1_eda'
) -> pd.DataFrame:
    """Analyze FICO score in bins with approval rates.

    Args:
        df: DataFrame with data
        target: Target variable (default: 'Approved')
        output_dir: Directory to save outputs

    Returns:
        DataFrame with approval rates by FICO bin
    """
    # Define FICO bins
    bins = [300, 580, 670, 740, 800, 850]
    labels = ['<580 (Poor)', '580-669 (Fair)', '670-739 (Good)', '740-799 (Very Good)', '800+ (Excellent)']

    # Create FICO bins
    df_copy = df.copy()
    df_copy['FICO_Bin'] = pd.cut(df_copy['FICO_score'], bins=bins, labels=labels, include_lowest=True)

    # Calculate approval rate by bin
    approval_by_fico = df_copy.groupby('FICO_Bin', observed=True)[target].agg([
        ('Total_Applications', 'count'),
        ('Total_Approvals', 'sum'),
        ('Approval_Rate', 'mean')
    ]).reset_index()

    # Save table
    Path(f"{output_dir}/tables").mkdir(parents=True, exist_ok=True)
    table_path = f"{output_dir}/tables/approval_by_fico_bins.csv"
    approval_by_fico.to_csv(table_path, index=False)
    print(f"✅ Saved: {table_path}")

    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(approval_by_fico))
    width = 0.35

    ax.bar([i - width/2 for i in x], approval_by_fico['Total_Applications']/1000,
           width, label='Applications (thousands)', color='lightblue', alpha=0.8)

    ax2 = ax.twinx()
    ax2.plot(x, approval_by_fico['Approval_Rate'], color='red', marker='o',
             linewidth=2, markersize=8, label='Approval Rate')

    ax.set_xlabel('FICO Score Range')
    ax.set_ylabel('Applications (thousands)', color='blue')
    ax2.set_ylabel('Approval Rate', color='red')
    ax.set_xticks(x)
    ax.set_xticklabels(approval_by_fico['FICO_Bin'], rotation=15, ha='right')
    ax.set_title('FICO Score Analysis: Applications vs Approval Rate')

    # Add value labels
    for i, v in enumerate(approval_by_fico['Approval_Rate']):
        ax2.text(i, v + 0.01, f'{v:.1%}', ha='center', va='bottom', color='red', fontweight='bold')

    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fig_path = f"{output_dir}/figures/approval_by_fico_bins.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {fig_path}")

    return approval_by_fico


def generate_univariate_summary(
    df: pd.DataFrame,
    output_dir: str = 'reports/phase1_eda'
) -> Dict:
    """Generate comprehensive univariate analysis summary.

    Args:
        df: DataFrame with data
        output_dir: Directory to save outputs

    Returns:
        Dictionary with all analysis results
    """
    import time
    start_time = time.time()

    print("\n" + "="*80)
    print("  PHASE 1.1: UNIVARIATE ANALYSIS")
    print("="*80 + "\n")

    results = {}

    # Overall approval rate
    overall_approval = df['Approved'].mean()
    print(f"📊 Overall Approval Rate: {overall_approval:.2%}\n")
    results['overall_approval_rate'] = overall_approval

    # Analyze categorical variables
    print("🔍 Analyzing Categorical Variables...")
    categorical_vars = ['Reason', 'Employment_Status', 'Employment_Sector', 'Lender', 'Fico_Score_group']

    for var in categorical_vars:
        if var in df.columns:
            print(f"\n   Analyzing: {var}")
            approval_table = analyze_categorical_variable(df, var, output_dir=output_dir)
            results[var] = approval_table

    # Analyze numerical variables
    print("\n🔍 Analyzing Numerical Variables...")
    numerical_vars = ['FICO_score', 'Loan_Amount', 'Monthly_Gross_Income', 'Monthly_Housing_Payment']

    numerical_stats = []
    for var in numerical_vars:
        if var in df.columns:
            print(f"\n   Analyzing: {var}")
            stats = analyze_numerical_variable(df, var, output_dir=output_dir)
            numerical_stats.append(stats)

    # Save numerical statistics
    stats_df = pd.DataFrame(numerical_stats)
    stats_path = f"{output_dir}/tables/numerical_variable_statistics.csv"
    Path(f"{output_dir}/tables").mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(stats_path, index=False)
    print(f"\n✅ Saved: {stats_path}")
    results['numerical_stats'] = stats_df

    # Analyze FICO bins
    print("\n🔍 Analyzing FICO Score Bins...")
    fico_bins = analyze_fico_bins(df, output_dir=output_dir)
    results['fico_bins'] = fico_bins

    # Generate key insights
    print("\n" + "="*80)
    print("  KEY INSIGHTS FROM UNIVARIATE ANALYSIS")
    print("="*80 + "\n")

    # FICO insight
    if 'fico_bins' in results:
        best_fico = fico_bins.loc[fico_bins['Approval_Rate'].idxmax()]
        worst_fico = fico_bins.loc[fico_bins['Approval_Rate'].idxmin()]
        print(f"💡 FICO Score Impact:")
        print(f"   - Highest approval: {best_fico['FICO_Bin']} ({best_fico['Approval_Rate']:.1%})")
        print(f"   - Lowest approval: {worst_fico['FICO_Bin']} ({worst_fico['Approval_Rate']:.1%})")
        print(f"   - Range: {(best_fico['Approval_Rate'] - worst_fico['Approval_Rate']):.1%} difference\n")

    # Reason insight
    if 'Reason' in results:
        reason_df = results['Reason']
        best_reason = reason_df.loc[reason_df['Approval_Rate'].idxmax()]
        worst_reason = reason_df.loc[reason_df['Approval_Rate'].idxmin()]
        print(f"💡 Loan Reason Impact:")
        print(f"   - Highest approval: {best_reason['Reason']} ({best_reason['Approval_Rate']:.1%})")
        print(f"   - Lowest approval: {worst_reason['Reason']} ({worst_reason['Approval_Rate']:.1%})\n")

    # Employment insight
    if 'Employment_Status' in results:
        emp_df = results['Employment_Status']
        best_emp = emp_df.loc[emp_df['Approval_Rate'].idxmax()]
        worst_emp = emp_df.loc[emp_df['Approval_Rate'].idxmin()]
        print(f"💡 Employment Status Impact:")
        print(f"   - Highest approval: {best_emp['Employment_Status']} ({best_emp['Approval_Rate']:.1%})")
        print(f"   - Lowest approval: {worst_emp['Employment_Status']} ({worst_emp['Approval_Rate']:.1%})\n")

    # Lender insight
    if 'Lender' in results:
        lender_df = results['Lender']
        print(f"💡 Lender Approval Rates:")
        for _, row in lender_df.iterrows():
            print(f"   - Lender {row['Lender']}: {row['Approval_Rate']:.1%} ({row['Total_Approvals']:,} / {row['Total_Applications']:,})")

    print("\n" + "="*80)
    print("  PHASE 1.1 COMPLETE")
    print("="*80 + "\n")

    # Build structured JSON output for UI integration
    execution_time = time.time() - start_time

    # Extract key insights
    insights = []
    if 'fico_bins' in results:
        best_fico = fico_bins.loc[fico_bins['Approval_Rate'].idxmax()]
        worst_fico = fico_bins.loc[fico_bins['Approval_Rate'].idxmin()]
        insights.append(f"FICO 800+: {best_fico['Approval_Rate']:.1%} approval vs <580: {worst_fico['Approval_Rate']:.1%}")

    if 'Lender' in results:
        lender_df = results['Lender']
        lender_c = lender_df[lender_df['Lender'] == 'C']['Approval_Rate'].values[0]
        insights.append(f"Lender C most lenient ({lender_c:.1%} approval)")

    if 'Employment_Status' in results:
        emp_df = results['Employment_Status']
        best_emp = emp_df.loc[emp_df['Approval_Rate'].idxmax()]
        worst_emp = emp_df.loc[emp_df['Approval_Rate'].idxmin()]
        diff = best_emp['Approval_Rate'] / worst_emp['Approval_Rate']
        insights.append(f"Employment: {best_emp['Approval_Rate']:.1%} vs {worst_emp['Approval_Rate']:.1%} ({diff:.1f}x difference)")

    # Build output structure
    json_output = {
        "success": True,
        "subphase": "Phase 1.1: Univariate Analysis",
        "summary": {
            "overall_approval_rate": float(overall_approval),
            "variables_analyzed": {
                "categorical": len(categorical_vars),
                "numerical": len(numerical_vars)
            }
        },
        "insights": insights,
        "outputs": {
            "tables": [
                "approval_by_reason.csv",
                "approval_by_employment_status.csv",
                "approval_by_employment_sector.csv",
                "approval_by_lender.csv",
                "approval_by_fico_score_group.csv",
                "approval_by_fico_bins.csv",
                "numerical_variable_statistics.csv"
            ],
            "figures": [
                "approval_by_reason.png",
                "approval_by_employment_status.png",
                "approval_by_employment_sector.png",
                "approval_by_lender.png",
                "approval_by_fico_score_group.png",
                "approval_by_fico_bins.png",
                "fico_score_by_approval.png",
                "loan_amount_by_approval.png",
                "monthly_gross_income_by_approval.png",
                "monthly_housing_payment_by_approval.png"
            ]
        },
        "execution_time": round(execution_time, 2)
    }

    # Output JSON marker for UI parsing
    import json
    print("\n__JSON_OUTPUT__")
    print(json.dumps(json_output, indent=2))

    return results


if __name__ == "__main__":
    # Test the module
    import sys
    sys.path.insert(0, '.')

    from src.phase1_eda.data_loader import load_and_validate

    print("Loading data...")
    df, _ = load_and_validate()

    print("\nRunning univariate analysis...")
    results = generate_univariate_summary(df)

    print("\n✅ Univariate analysis complete!")
    print(f"   - Outputs saved to: reports/phase1_eda/")
    print(f"   - Figures: reports/phase1_eda/figures/")
    print(f"   - Tables: reports/phase1_eda/tables/")
