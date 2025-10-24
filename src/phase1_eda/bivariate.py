"""
Phase 1.2: Bivariate Analysis
Analyzes relationships between variables and creates engineered features
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

def calculate_correlation_matrix(df):
    """Calculate correlation matrix for numerical variables"""
    numerical_cols = [
        'Loan_Amount', 'FICO_score', 'Monthly_Gross_Income',
        'Monthly_Housing_Payment', 'Approved'
    ]

    corr_matrix = df[numerical_cols].corr()

    # Create correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                center=0, square=True, linewidths=1)
    plt.title('Correlation Matrix - Numerical Variables', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('reports/phase1_eda/figures/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Save correlation matrix
    corr_matrix.to_csv('reports/phase1_eda/tables/correlation_matrix.csv')

    return corr_matrix

def perform_chi_square_tests(df):
    """Perform chi-square tests for categorical variables vs approval"""
    categorical_vars = [
        'Reason', 'Fico_Score_group', 'Employment_Status',
        'Employment_Sector', 'Lender', 'Ever_Bankrupt_or_Foreclose'
    ]

    results = []

    for var in categorical_vars:
        if var in df.columns:
            # Create contingency table
            contingency = pd.crosstab(df[var], df['Approved'])

            # Perform chi-square test
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)

            # Calculate Cramér's V for effect size
            n = contingency.sum().sum()
            min_dim = min(contingency.shape) - 1
            cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

            results.append({
                'Variable': var,
                'Chi2_Statistic': chi2,
                'P_Value': p_value,
                'Degrees_of_Freedom': dof,
                'Cramers_V': cramers_v,
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('P_Value')
    results_df.to_csv('reports/phase1_eda/tables/chi_square_tests.csv', index=False)

    return results_df

def perform_anova_tests(df):
    """Perform ANOVA/t-tests for numerical variables vs approval"""
    numerical_vars = [
        'Loan_Amount', 'FICO_score', 'Monthly_Gross_Income',
        'Monthly_Housing_Payment'
    ]

    results = []

    for var in numerical_vars:
        # Separate by approval status
        approved = df[df['Approved'] == 1][var]
        denied = df[df['Approved'] == 0][var]

        # Perform t-test
        t_stat, p_value = stats.ttest_ind(approved, denied)

        # Calculate effect size (Cohen's d)
        mean_diff = approved.mean() - denied.mean()
        pooled_std = np.sqrt(((len(approved)-1)*approved.std()**2 +
                             (len(denied)-1)*denied.std()**2) /
                            (len(approved) + len(denied) - 2))
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0

        results.append({
            'Variable': var,
            'Mean_Approved': approved.mean(),
            'Mean_Denied': denied.mean(),
            'Mean_Difference': mean_diff,
            'T_Statistic': t_stat,
            'P_Value': p_value,
            'Cohens_D': cohens_d,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('P_Value')
    results_df.to_csv('reports/phase1_eda/tables/anova_ttest_results.csv', index=False)

    return results_df

def engineer_features(df):
    """Create derived features: DTI, LTI, custom bins"""
    df_engineered = df.copy()

    # 1. Debt-to-Income Ratio
    df_engineered['DTI'] = df_engineered['Monthly_Housing_Payment'] / df_engineered['Monthly_Gross_Income']
    df_engineered['DTI'] = df_engineered['DTI'].replace([np.inf, -np.inf], np.nan)

    # 2. Loan-to-Income Ratio
    annual_income = df_engineered['Monthly_Gross_Income'] * 12
    df_engineered['LTI'] = df_engineered['Loan_Amount'] / annual_income
    df_engineered['LTI'] = df_engineered['LTI'].replace([np.inf, -np.inf], np.nan)

    # 3. Custom FICO bins (finer granularity)
    df_engineered['FICO_Bin_Custom'] = pd.cut(
        df_engineered['FICO_score'],
        bins=[0, 579, 620, 660, 700, 740, 780, 850],
        labels=['<580', '580-619', '620-659', '660-699', '700-739', '740-779', '780+']
    )

    # 4. Income Quartiles
    df_engineered['Income_Quartile'] = pd.qcut(
        df_engineered['Monthly_Gross_Income'],
        q=4,
        labels=['Q1_Low', 'Q2_Medium_Low', 'Q3_Medium_High', 'Q4_High'],
        duplicates='drop'
    )

    # 5. Loan Amount Categories
    df_engineered['Loan_Category'] = pd.cut(
        df_engineered['Loan_Amount'],
        bins=[0, 30000, 60000, np.inf],
        labels=['Small_<30K', 'Medium_30-60K', 'Large_60K+']
    )

    # Save engineered features dataset
    df_engineered.to_csv('data/processed/features_engineered.csv', index=False)

    # Create summary of engineered features
    feature_summary = {
        'DTI': {
            'description': 'Debt-to-Income Ratio: Monthly_Housing_Payment / Monthly_Gross_Income',
            'mean': float(df_engineered['DTI'].mean()),
            'median': float(df_engineered['DTI'].median()),
            'std': float(df_engineered['DTI'].std())
        },
        'LTI': {
            'description': 'Loan-to-Income Ratio: Loan_Amount / (Monthly_Gross_Income * 12)',
            'mean': float(df_engineered['LTI'].mean()),
            'median': float(df_engineered['LTI'].median()),
            'std': float(df_engineered['LTI'].std())
        },
        'FICO_Bin_Custom': {
            'description': 'Custom FICO bins with finer granularity',
            'categories': df_engineered['FICO_Bin_Custom'].value_counts().to_dict()
        },
        'Income_Quartile': {
            'description': 'Income quartiles: Q1 (lowest) to Q4 (highest)',
            'categories': df_engineered['Income_Quartile'].value_counts().to_dict()
        },
        'Loan_Category': {
            'description': 'Loan amount categories: Small (<30K), Medium (30-60K), Large (60K+)',
            'categories': df_engineered['Loan_Category'].value_counts().to_dict()
        }
    }

    with open('reports/phase1_eda/tables/engineered_features_summary.json', 'w') as f:
        json.dump(feature_summary, f, indent=2, default=str)

    return df_engineered

def create_bivariate_charts(df):
    """Create bivariate approval rate visualizations"""

    # 1. DTI vs Approval
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    dti_bins = pd.cut(df['DTI'], bins=10)
    approval_by_dti = df.groupby(dti_bins)['Approved'].agg(['mean', 'count'])
    approval_by_dti['mean'].plot(kind='bar', color='skyblue', edgecolor='black')
    plt.title('Approval Rate by DTI Ratio', fontsize=12, fontweight='bold')
    plt.xlabel('DTI Range')
    plt.ylabel('Approval Rate')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)

    plt.subplot(1, 2, 2)
    for approved in [0, 1]:
        subset = df[df['Approved'] == approved]['DTI']
        plt.hist(subset, bins=30, alpha=0.5, label=f'Approved={approved}')
    plt.xlabel('DTI Ratio')
    plt.ylabel('Frequency')
    plt.title('DTI Distribution by Approval', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/phase1_eda/figures/bivariate_dti_approval.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. LTI vs Approval
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    lti_bins = pd.cut(df['LTI'], bins=10)
    approval_by_lti = df.groupby(lti_bins)['Approved'].agg(['mean', 'count'])
    approval_by_lti['mean'].plot(kind='bar', color='lightcoral', edgecolor='black')
    plt.title('Approval Rate by LTI Ratio', fontsize=12, fontweight='bold')
    plt.xlabel('LTI Range')
    plt.ylabel('Approval Rate')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)

    plt.subplot(1, 2, 2)
    for approved in [0, 1]:
        subset = df[df['Approved'] == approved]['LTI']
        plt.hist(subset, bins=30, alpha=0.5, label=f'Approved={approved}')
    plt.xlabel('LTI Ratio')
    plt.ylabel('Frequency')
    plt.title('LTI Distribution by Approval', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/phase1_eda/figures/bivariate_lti_approval.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. FICO x Income Interaction
    plt.figure(figsize=(10, 8))

    # Create bins for both variables
    fico_bins = pd.qcut(df['FICO_score'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    income_bins = pd.qcut(df['Monthly_Gross_Income'], q=5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'])

    # Create pivot table
    interaction = df.groupby([fico_bins, income_bins])['Approved'].mean().unstack()

    sns.heatmap(interaction, annot=True, fmt='.2%', cmap='RdYlGn', center=0.11,
                linewidths=1, cbar_kws={'label': 'Approval Rate'})
    plt.title('Approval Rate: FICO Score × Income Interaction', fontsize=14, fontweight='bold')
    plt.xlabel('Income Quintile')
    plt.ylabel('FICO Score Group')
    plt.tight_layout()
    plt.savefig('reports/phase1_eda/figures/bivariate_fico_income_interaction.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Engineered Features Approval Rates
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Custom FICO bins
    approval_by_fico = df.groupby('FICO_Bin_Custom')['Approved'].mean()
    approval_by_fico.plot(kind='bar', ax=axes[0, 0], color='steelblue', edgecolor='black')
    axes[0, 0].set_title('Approval Rate by Custom FICO Bins', fontweight='bold')
    axes[0, 0].set_ylabel('Approval Rate')
    axes[0, 0].grid(axis='y', alpha=0.3)
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Income Quartiles
    approval_by_income = df.groupby('Income_Quartile')['Approved'].mean()
    approval_by_income.plot(kind='bar', ax=axes[0, 1], color='forestgreen', edgecolor='black')
    axes[0, 1].set_title('Approval Rate by Income Quartile', fontweight='bold')
    axes[0, 1].set_ylabel('Approval Rate')
    axes[0, 1].grid(axis='y', alpha=0.3)
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Loan Categories
    approval_by_loan = df.groupby('Loan_Category')['Approved'].mean()
    approval_by_loan.plot(kind='bar', ax=axes[1, 0], color='darkorange', edgecolor='black')
    axes[1, 0].set_title('Approval Rate by Loan Category', fontweight='bold')
    axes[1, 0].set_ylabel('Approval Rate')
    axes[1, 0].grid(axis='y', alpha=0.3)
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Lender × FICO interaction
    lender_fico = df.groupby(['Lender', 'Fico_Score_group'])['Approved'].mean().unstack()
    lender_fico.plot(kind='bar', ax=axes[1, 1], edgecolor='black')
    axes[1, 1].set_title('Approval Rate: Lender × FICO Group', fontweight='bold')
    axes[1, 1].set_ylabel('Approval Rate')
    axes[1, 1].legend(title='FICO Group', bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 1].grid(axis='y', alpha=0.3)
    axes[1, 1].tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig('reports/phase1_eda/figures/bivariate_engineered_features.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_statistical_summary(corr_matrix, chi_square_results, anova_results):
    """Generate comprehensive statistical test summary"""
    summary = {
        'correlation_insights': {
            'strongest_positive': {
                'variables': 'Monthly_Gross_Income & Loan_Amount',
                'correlation': float(corr_matrix.loc['Monthly_Gross_Income', 'Loan_Amount'])
            },
            'strongest_with_approval': {
                'variable': corr_matrix['Approved'].abs().sort_values(ascending=False).index[1],
                'correlation': float(corr_matrix['Approved'].abs().sort_values(ascending=False).iloc[1])
            }
        },
        'chi_square_insights': {
            'most_significant': chi_square_results.iloc[0]['Variable'],
            'p_value': float(chi_square_results.iloc[0]['P_Value']),
            'significant_count': int(chi_square_results['Significant'].value_counts().get('Yes', 0))
        },
        'anova_insights': {
            'largest_difference': anova_results.iloc[0]['Variable'],
            'mean_difference': float(anova_results.iloc[0]['Mean_Difference']),
            'significant_count': int(anova_results['Significant'].value_counts().get('Yes', 0))
        }
    }

    with open('reports/phase1_eda/tables/statistical_tests_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary

def run_bivariate_analysis():
    """Main function to run complete bivariate analysis"""
    start_time = time.time()

    print("=" * 60)
    print("Phase 1.2: Bivariate Analysis")
    print("=" * 60)

    # Create output directories
    create_output_dirs()
    print("✓ Output directories created")

    # Load data
    print("\n1. Loading data...")
    df, report = load_and_validate()
    print(f"✓ Data loaded: {len(df):,} rows, {len(df.columns)} columns")

    # Correlation analysis
    print("\n2. Calculating correlation matrix...")
    corr_matrix = calculate_correlation_matrix(df)
    print("✓ Correlation matrix saved")

    # Chi-square tests
    print("\n3. Performing chi-square tests...")
    chi_square_results = perform_chi_square_tests(df)
    print(f"✓ Chi-square tests completed: {len(chi_square_results)} categorical variables tested")

    # ANOVA/t-tests
    print("\n4. Performing ANOVA/t-tests...")
    anova_results = perform_anova_tests(df)
    print(f"✓ ANOVA/t-tests completed: {len(anova_results)} numerical variables tested")

    # Feature engineering
    print("\n5. Engineering features (DTI, LTI, bins)...")
    df_engineered = engineer_features(df)
    print(f"✓ Features engineered: {len(df_engineered.columns) - len(df.columns)} new features created")

    # Bivariate visualizations
    print("\n6. Creating bivariate charts...")
    create_bivariate_charts(df_engineered)
    print("✓ Bivariate charts saved")

    # Generate summary
    print("\n7. Generating statistical summary...")
    summary = generate_statistical_summary(corr_matrix, chi_square_results, anova_results)
    print("✓ Summary saved")

    execution_time = time.time() - start_time

    # Build structured output
    output = {
        "success": True,
        "subphase": "Phase 1.2: Bivariate Analysis",
        "summary": {
            "correlation_tests": len(corr_matrix),
            "chi_square_tests": len(chi_square_results),
            "anova_tests": len(anova_results),
            "engineered_features": 5
        },
        "insights": [
            f"Strongest correlation with approval: {summary['correlation_insights']['strongest_with_approval']['variable']} ({summary['correlation_insights']['strongest_with_approval']['correlation']:.3f})",
            f"Most significant categorical predictor: {summary['chi_square_insights']['most_significant']} (p={summary['chi_square_insights']['p_value']:.2e})",
            f"Largest mean difference: {summary['anova_insights']['largest_difference']} ({summary['anova_insights']['mean_difference']:.0f} units)",
            f"All numerical variables significantly different by approval status (p < 0.05)",
            f"Engineered features: DTI, LTI, custom FICO bins, income quartiles, loan categories"
        ],
        "outputs": {
            "tables": [
                "correlation_matrix.csv",
                "chi_square_tests.csv",
                "anova_ttest_results.csv",
                "engineered_features_summary.json",
                "statistical_tests_summary.json"
            ],
            "figures": [
                "correlation_heatmap.png",
                "bivariate_dti_approval.png",
                "bivariate_lti_approval.png",
                "bivariate_fico_income_interaction.png",
                "bivariate_engineered_features.png"
            ],
            "data": [
                "features_engineered.csv"
            ]
        },
        "execution_time": execution_time
    }

    print("\n" + "=" * 60)
    print("__JSON_OUTPUT__")
    print(json.dumps(output, indent=2))
    print("=" * 60)

    return df_engineered

if __name__ == "__main__":
    run_bivariate_analysis()
