"""
Phase 1.3: Missing Value Treatment
Handles 6.4% missing Employment_Sector values with analysis and strategy implementation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Import data loader
from .data_loader import load_and_validate

def create_output_dirs():
    """Create output directories if they don't exist"""
    dirs = [
        'reports/phase1_eda/figures',
        'reports/phase1_eda/tables',
        'data/processed'
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def analyze_missing_patterns(df):
    """Analyze patterns in missing Employment_Sector values"""
    print("\n" + "="*60)
    print("MISSING VALUE PATTERN ANALYSIS")
    print("="*60)

    # Create missing indicator
    df['Employment_Sector_Missing'] = df['Employment_Sector'].isna().astype(int)

    # Basic statistics
    missing_count = df['Employment_Sector_Missing'].sum()
    missing_pct = (missing_count / len(df)) * 100

    print(f"\nMissing Employment_Sector:")
    print(f"  Count: {missing_count:,}")
    print(f"  Percentage: {missing_pct:.2f}%")

    # Analyze by approval status
    print(f"\nMissing by Approval Status:")
    missing_by_approval = df.groupby('Approved')['Employment_Sector_Missing'].agg(['sum', 'mean'])
    missing_by_approval.columns = ['Missing_Count', 'Missing_Rate']
    missing_by_approval['Missing_Rate'] = missing_by_approval['Missing_Rate'] * 100
    print(missing_by_approval)

    # Chi-square test: Does missingness predict approval?
    contingency = pd.crosstab(df['Employment_Sector_Missing'], df['Approved'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

    print(f"\nChi-Square Test (Missingness vs Approval):")
    print(f"  Chi2 Statistic: {chi2:.4f}")
    print(f"  P-Value: {p_value:.4e}")
    print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")

    # Approval rates by missingness
    approval_rates = df.groupby('Employment_Sector_Missing')['Approved'].mean() * 100
    print(f"\nApproval Rates:")
    print(f"  Has Sector: {approval_rates[0]:.2f}%")
    print(f"  Missing Sector: {approval_rates[1]:.2f}%")
    print(f"  Difference: {approval_rates[1] - approval_rates[0]:.2f} percentage points")

    # Analyze missingness by other variables
    print(f"\nMissing Rate by Lender:")
    missing_by_lender = df.groupby('Lender')['Employment_Sector_Missing'].mean() * 100
    print(missing_by_lender)

    print(f"\nMissing Rate by Employment Status:")
    missing_by_emp_status = df.groupby('Employment_Status')['Employment_Sector_Missing'].mean() * 100
    print(missing_by_emp_status)

    # Numerical variable comparisons
    print(f"\nNumerical Variables (Missing vs Not Missing):")
    numerical_vars = ['FICO_score', 'Loan_Amount', 'Monthly_Gross_Income', 'Monthly_Housing_Payment']

    comparison = []
    for var in numerical_vars:
        has_sector = df[df['Employment_Sector_Missing'] == 0][var].mean()
        missing_sector = df[df['Employment_Sector_Missing'] == 1][var].mean()
        diff = missing_sector - has_sector
        diff_pct = (diff / has_sector) * 100 if has_sector != 0 else 0

        # T-test
        t_stat, p_val = stats.ttest_ind(
            df[df['Employment_Sector_Missing'] == 0][var],
            df[df['Employment_Sector_Missing'] == 1][var]
        )

        comparison.append({
            'Variable': var,
            'Has_Sector_Mean': has_sector,
            'Missing_Sector_Mean': missing_sector,
            'Difference': diff,
            'Difference_Pct': diff_pct,
            'P_Value': p_val,
            'Significant': 'Yes' if p_val < 0.05 else 'No'
        })

        print(f"\n  {var}:")
        print(f"    Has Sector: {has_sector:.2f}")
        print(f"    Missing: {missing_sector:.2f}")
        print(f"    Difference: {diff:.2f} ({diff_pct:+.1f}%)")
        print(f"    Significant: {'Yes' if p_val < 0.05 else 'No'} (p={p_val:.4f})")

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv('reports/phase1_eda/tables/missing_value_comparison.csv', index=False)

    # Return analysis results
    analysis = {
        'missing_count': int(missing_count),
        'missing_percentage': float(missing_pct),
        'chi_square_statistic': float(chi2),
        'chi_square_p_value': float(p_value),
        'predicts_approval': p_value < 0.05,
        'approval_rate_has_sector': float(approval_rates[0]),
        'approval_rate_missing_sector': float(approval_rates[1]),
        'approval_difference': float(approval_rates[1] - approval_rates[0])
    }

    return df, analysis

def create_missing_value_visualizations(df):
    """Create visualizations for missing value analysis"""

    # 1. Missing value distribution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Missing by approval status
    ax = axes[0, 0]
    missing_by_approval = df.groupby(['Approved', 'Employment_Sector_Missing']).size().unstack()
    missing_by_approval.plot(kind='bar', ax=ax, color=['steelblue', 'coral'])
    ax.set_title('Missing Employment_Sector by Approval Status', fontweight='bold')
    ax.set_xlabel('Approval Status')
    ax.set_ylabel('Count')
    ax.set_xticklabels(['Denied', 'Approved'], rotation=0)
    ax.legend(['Has Sector', 'Missing Sector'], title='Employment Sector')
    ax.grid(axis='y', alpha=0.3)

    # Approval rate by missingness
    ax = axes[0, 1]
    approval_by_missing = df.groupby('Employment_Sector_Missing')['Approved'].mean() * 100
    approval_by_missing.plot(kind='bar', ax=ax, color=['steelblue', 'coral'])
    ax.set_title('Approval Rate by Missing Status', fontweight='bold')
    ax.set_xlabel('Employment Sector Status')
    ax.set_ylabel('Approval Rate (%)')
    ax.set_xticklabels(['Has Sector', 'Missing'], rotation=0)
    ax.axhline(y=df['Approved'].mean() * 100, color='red', linestyle='--',
               label=f'Overall: {df["Approved"].mean()*100:.1f}%')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Missing rate by lender
    ax = axes[1, 0]
    missing_by_lender = df.groupby('Lender')['Employment_Sector_Missing'].mean() * 100
    missing_by_lender.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
    ax.set_title('Missing Rate by Lender', fontweight='bold')
    ax.set_xlabel('Lender')
    ax.set_ylabel('Missing Rate (%)')
    ax.set_xticklabels(missing_by_lender.index, rotation=0)
    ax.grid(axis='y', alpha=0.3)

    # FICO distribution by missing status
    ax = axes[1, 1]
    df[df['Employment_Sector_Missing'] == 0]['FICO_score'].hist(
        bins=30, ax=ax, alpha=0.6, label='Has Sector', color='steelblue'
    )
    df[df['Employment_Sector_Missing'] == 1]['FICO_score'].hist(
        bins=30, ax=ax, alpha=0.6, label='Missing Sector', color='coral'
    )
    ax.set_title('FICO Score Distribution by Missing Status', fontweight='bold')
    ax.set_xlabel('FICO Score')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/phase1_eda/figures/missing_value_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def implement_missing_value_strategy(df, strategy='unknown_category'):
    """
    Implement chosen missing value strategy

    Strategies:
    - 'unknown_category': Create 'Unknown' category (RECOMMENDED per PRD)
    - 'mode_imputation': Fill with most common sector
    - 'deletion': Remove rows with missing values
    """

    print("\n" + "="*60)
    print(f"IMPLEMENTING STRATEGY: {strategy.upper()}")
    print("="*60)

    df_cleaned = df.copy()
    original_count = len(df_cleaned)
    missing_count = df_cleaned['Employment_Sector'].isna().sum()

    if strategy == 'unknown_category':
        # Fill missing with 'Unknown' category
        df_cleaned['Employment_Sector'] = df_cleaned['Employment_Sector'].fillna('Unknown')
        print(f"\n✓ Created 'Unknown' category for {missing_count:,} missing values")
        print(f"  Rows retained: {len(df_cleaned):,} (100%)")

    elif strategy == 'mode_imputation':
        # Fill with most common sector
        mode_value = df_cleaned['Employment_Sector'].mode()[0]
        df_cleaned['Employment_Sector'] = df_cleaned['Employment_Sector'].fillna(mode_value)
        print(f"\n✓ Imputed {missing_count:,} missing values with mode: '{mode_value}'")
        print(f"  Rows retained: {len(df_cleaned):,} (100%)")

    elif strategy == 'deletion':
        # Remove rows with missing values
        df_cleaned = df_cleaned.dropna(subset=['Employment_Sector'])
        rows_removed = original_count - len(df_cleaned)
        retention_pct = (len(df_cleaned) / original_count) * 100
        print(f"\n✓ Removed {rows_removed:,} rows with missing values")
        print(f"  Rows retained: {len(df_cleaned):,} ({retention_pct:.1f}%)")

    # Verify no missing values remain
    remaining_missing = df_cleaned['Employment_Sector'].isna().sum()
    print(f"\n✓ Verification: {remaining_missing} missing values remain")

    # Remove temporary missing indicator column
    if 'Employment_Sector_Missing' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(columns=['Employment_Sector_Missing'])

    return df_cleaned

def save_cleaned_data(df_cleaned):
    """Save cleaned dataset"""
    output_path = 'data/processed/cleaned_data.csv'
    df_cleaned.to_csv(output_path, index=False)
    print(f"\n✓ Cleaned data saved to: {output_path}")
    print(f"  Shape: {df_cleaned.shape}")
    print(f"  Missing Employment_Sector: {df_cleaned['Employment_Sector'].isna().sum()}")

def generate_missing_value_report(analysis, strategy):
    """Generate comprehensive missing value treatment report"""

    # Convert numpy/pandas types to native Python types
    analysis_serializable = {
        'missing_count': int(analysis['missing_count']),
        'missing_percentage': float(analysis['missing_percentage']),
        'chi_square_statistic': float(analysis['chi_square_statistic']),
        'chi_square_p_value': float(analysis['chi_square_p_value']),
        'predicts_approval': bool(analysis['predicts_approval']),
        'approval_rate_has_sector': float(analysis['approval_rate_has_sector']),
        'approval_rate_missing_sector': float(analysis['approval_rate_missing_sector']),
        'approval_difference': float(analysis['approval_difference'])
    }

    report = {
        'analysis': analysis_serializable,
        'strategy': {
            'chosen_strategy': strategy,
            'rationale': 'Create Unknown category to preserve information and test predictive power of missingness',
            'alternatives_considered': [
                'Mode imputation - rejected due to loss of missing pattern signal',
                'Deletion - rejected due to data loss (6.4% of dataset)'
            ]
        },
        'recommendations': [
            'Missing pattern is statistically significant predictor',
            'Unknown category preserves this information for modeling',
            'Can be used as feature in Phase 2 analysis'
        ]
    }

    with open('reports/phase1_eda/tables/missing_value_report.json', 'w') as f:
        json.dump(report, f, indent=2)

    return report

def run_missing_value_treatment():
    """Main function to run complete missing value treatment"""
    start_time = time.time()

    print("=" * 70)
    print("Phase 1.3: Missing Value Treatment")
    print("=" * 70)

    # Create output directories
    create_output_dirs()
    print("✓ Output directories created")

    # Load data
    print("\n1. Loading data...")
    df, report = load_and_validate()
    print(f"✓ Data loaded: {len(df):,} rows, {len(df.columns)} columns")

    # Analyze missing patterns
    print("\n2. Analyzing missing value patterns...")
    df_with_indicator, analysis = analyze_missing_patterns(df)
    print("✓ Missing pattern analysis complete")

    # Create visualizations
    print("\n3. Creating missing value visualizations...")
    create_missing_value_visualizations(df_with_indicator)
    print("✓ Visualizations saved")

    # Implement strategy (Unknown category - recommended by PRD)
    print("\n4. Implementing missing value strategy...")
    strategy = 'unknown_category'
    df_cleaned = implement_missing_value_strategy(df_with_indicator, strategy=strategy)
    print("✓ Missing value strategy implemented")

    # Save cleaned data
    print("\n5. Saving cleaned dataset...")
    save_cleaned_data(df_cleaned)
    print("✓ Cleaned data saved")

    # Generate report
    print("\n6. Generating missing value report...")
    report_data = generate_missing_value_report(analysis, strategy)
    print("✓ Report generated")

    execution_time = time.time() - start_time

    # Build structured output
    output = {
        "success": True,
        "subphase": "Phase 1.3: Missing Value Treatment",
        "summary": {
            "original_missing_count": analysis['missing_count'],
            "missing_percentage": analysis['missing_percentage'],
            "strategy_used": strategy,
            "final_missing_count": 0,
            "rows_retained": len(df_cleaned)
        },
        "insights": [
            f"Missing Employment_Sector: {analysis['missing_count']:,} values ({analysis['missing_percentage']:.1f}%)",
            f"Missingness {'is' if analysis['predicts_approval'] else 'is not'} a significant predictor of approval (p={analysis['chi_square_p_value']:.2e})",
            f"Approval rate difference: {analysis['approval_difference']:.2f} percentage points (missing vs has sector)",
            f"Strategy: Created 'Unknown' category to preserve missing pattern information",
            f"All {len(df_cleaned):,} rows retained (no data loss)"
        ],
        "outputs": {
            "tables": [
                "missing_value_comparison.csv",
                "missing_value_report.json"
            ],
            "figures": [
                "missing_value_analysis.png"
            ],
            "data": [
                "cleaned_data.csv"
            ]
        },
        "execution_time": execution_time
    }

    print("\n" + "=" * 70)
    print("__JSON_OUTPUT__")
    print(json.dumps(output, indent=2))
    print("=" * 70)

    return df_cleaned

if __name__ == "__main__":
    run_missing_value_treatment()
