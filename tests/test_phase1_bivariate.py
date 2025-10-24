"""
Validation Test for Phase 1.2: Bivariate Analysis

Verifies all required outputs exist and meet quality standards per PRD requirements.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path


class ValidationResults:
    """Store and format validation results"""
    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def add_check(self, category, name, passed, message):
        self.checks.append({
            'category': category,
            'name': name,
            'passed': passed,
            'message': message
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def add_warning(self, category, name, message):
        self.checks.append({
            'category': category,
            'name': name,
            'passed': True,
            'message': f'⚠️ {message}'
        })
        self.warnings += 1

    def get_summary(self):
        total = self.passed + self.failed
        pass_rate = self.passed / total if total > 0 else 0
        return {
            'total_checks': total,
            'passed': self.passed,
            'failed': self.failed,
            'warnings': self.warnings,
            'pass_rate': pass_rate
        }


def validate_phase1_bivariate():
    """
    Validate Phase 1.2 outputs according to PRD section 1.2 validation criteria

    Expected outputs to verify:
    - reports/phase1_eda/figures/correlation_heatmap.png exists
    - reports/phase1_eda/tables/statistical_tests.csv exists (chi-square, t-test results)
    - data/processed/features_engineered.csv exists with DTI, LTI columns
    - All p-values calculated and significant relationships identified
    - Bivariate charts show variable interactions with approval
    """
    print("=" * 70)
    print("PHASE 1.2: BIVARIATE ANALYSIS - VALIDATION TEST")
    print("=" * 70)

    results = ValidationResults()

    # ========================================================================
    # CATEGORY 1: FIGURES - Required visualizations
    # ========================================================================
    print("\n📊 Validating Figures...")

    required_figures = [
        'correlation_heatmap.png',
        'bivariate_dti_approval.png',
        'bivariate_lti_approval.png',
        'bivariate_fico_income_interaction.png',
        'bivariate_engineered_features.png'
    ]

    figures_dir = Path('reports/phase1_eda/figures')

    for fig in required_figures:
        fig_path = figures_dir / fig
        exists = fig_path.exists()
        size = fig_path.stat().st_size if exists else 0

        if exists and size > 10000:  # At least 10KB
            results.add_check('Figures', fig, True, f'✅ Found ({size/1024:.1f}KB)')
        elif exists:
            results.add_warning('Figures', fig, f'File too small ({size} bytes)')
        else:
            results.add_check('Figures', fig, False, '❌ Not found')

    # ========================================================================
    # CATEGORY 2: TABLES - Statistical test results
    # ========================================================================
    print("\n📋 Validating Tables...")

    required_tables = {
        'correlation_matrix.csv': {'min_rows': 4, 'min_cols': 4},
        'chi_square_tests.csv': {'min_rows': 5, 'required_cols': ['Variable', 'P_Value', 'Significant']},
        'anova_ttest_results.csv': {'min_rows': 4, 'required_cols': ['Variable', 'P_Value', 'Significant']},
        'engineered_features_summary.json': {},
        'statistical_tests_summary.json': {}
    }

    tables_dir = Path('reports/phase1_eda/tables')

    for table, criteria in required_tables.items():
        table_path = tables_dir / table

        if not table_path.exists():
            results.add_check('Tables', table, False, '❌ Not found')
            continue

        if table.endswith('.csv'):
            try:
                df = pd.read_csv(table_path)

                # Check minimum rows
                if 'min_rows' in criteria:
                    if len(df) >= criteria['min_rows']:
                        results.add_check('Tables', f'{table} (rows)', True,
                                        f'✅ {len(df)} rows')
                    else:
                        results.add_check('Tables', f'{table} (rows)', False,
                                        f'❌ Only {len(df)} rows, expected {criteria["min_rows"]}+')

                # Check required columns
                if 'required_cols' in criteria:
                    missing_cols = set(criteria['required_cols']) - set(df.columns)
                    if not missing_cols:
                        results.add_check('Tables', f'{table} (columns)', True,
                                        f'✅ All required columns present')
                    else:
                        results.add_check('Tables', f'{table} (columns)', False,
                                        f'❌ Missing columns: {missing_cols}')

                # Check minimum columns
                if 'min_cols' in criteria:
                    if len(df.columns) >= criteria['min_cols']:
                        results.add_check('Tables', f'{table} (columns)', True,
                                        f'✅ {len(df.columns)} columns')
                    else:
                        results.add_check('Tables', f'{table} (columns)', False,
                                        f'❌ Only {len(df.columns)} columns')

            except Exception as e:
                results.add_check('Tables', table, False, f'❌ Error reading: {str(e)}')

        elif table.endswith('.json'):
            try:
                with open(table_path, 'r') as f:
                    data = json.load(f)
                results.add_check('Tables', table, True, f'✅ Valid JSON')
            except Exception as e:
                results.add_check('Tables', table, False, f'❌ Invalid JSON: {str(e)}')

    # ========================================================================
    # CATEGORY 3: DATA - Engineered features dataset
    # ========================================================================
    print("\n💾 Validating Processed Data...")

    features_path = Path('data/processed/features_engineered.csv')

    if not features_path.exists():
        results.add_check('Data', 'features_engineered.csv', False, '❌ Not found')
    else:
        try:
            df = pd.read_csv(features_path)

            # Check row count
            if len(df) == 100000:
                results.add_check('Data', 'Row count', True, '✅ 100,000 rows preserved')
            else:
                results.add_check('Data', 'Row count', False,
                                f'❌ {len(df):,} rows (expected 100,000)')

            # Check engineered features exist
            required_features = ['DTI', 'LTI', 'FICO_Bin_Custom', 'Income_Quartile', 'Loan_Category']

            for feature in required_features:
                if feature in df.columns:
                    non_null = df[feature].notna().sum()
                    null_pct = (df[feature].isna().sum() / len(df)) * 100

                    if null_pct < 5:  # Less than 5% missing
                        results.add_check('Data', f'{feature} feature', True,
                                        f'✅ {non_null:,} non-null values ({100-null_pct:.1f}%)')
                    else:
                        results.add_warning('Data', f'{feature} feature',
                                          f'{null_pct:.1f}% missing values')
                else:
                    results.add_check('Data', f'{feature} feature', False, '❌ Column not found')

            # Validate DTI calculation
            if 'DTI' in df.columns and 'Monthly_Housing_Payment' in df.columns and 'Monthly_Gross_Income' in df.columns:
                sample = df.dropna(subset=['DTI', 'Monthly_Housing_Payment', 'Monthly_Gross_Income']).head(100)
                expected_dti = sample['Monthly_Housing_Payment'] / sample['Monthly_Gross_Income']
                actual_dti = sample['DTI']

                if np.allclose(expected_dti, actual_dti, rtol=0.01):
                    results.add_check('Data', 'DTI calculation', True, '✅ Correctly calculated')
                else:
                    results.add_check('Data', 'DTI calculation', False, '❌ Calculation error')

            # Validate LTI calculation
            if 'LTI' in df.columns and 'Loan_Amount' in df.columns and 'Monthly_Gross_Income' in df.columns:
                sample = df.dropna(subset=['LTI', 'Loan_Amount', 'Monthly_Gross_Income']).head(100)
                expected_lti = sample['Loan_Amount'] / (sample['Monthly_Gross_Income'] * 12)
                actual_lti = sample['LTI']

                if np.allclose(expected_lti, actual_lti, rtol=0.01):
                    results.add_check('Data', 'LTI calculation', True, '✅ Correctly calculated')
                else:
                    results.add_check('Data', 'LTI calculation', False, '❌ Calculation error')

        except Exception as e:
            results.add_check('Data', 'features_engineered.csv', False, f'❌ Error reading: {str(e)}')

    # ========================================================================
    # CATEGORY 4: CONTENT QUALITY - Statistical significance and insights
    # ========================================================================
    print("\n🔍 Validating Content Quality...")

    # Check chi-square results
    chi_path = tables_dir / 'chi_square_tests.csv'
    if chi_path.exists():
        try:
            chi_df = pd.read_csv(chi_path)

            # Check p-values present
            if 'P_Value' in chi_df.columns:
                p_values_present = chi_df['P_Value'].notna().all()
                if p_values_present:
                    results.add_check('Content', 'Chi-square p-values', True, '✅ All present')
                else:
                    results.add_check('Content', 'Chi-square p-values', False, '❌ Some missing')

            # Check significant relationships identified
            if 'Significant' in chi_df.columns:
                sig_count = (chi_df['Significant'] == 'Yes').sum()
                if sig_count > 0:
                    results.add_check('Content', 'Significant predictors', True,
                                    f'✅ {sig_count} significant variables found')
                else:
                    results.add_warning('Content', 'Significant predictors',
                                      'No significant variables found')
        except Exception as e:
            results.add_check('Content', 'Chi-square analysis', False, f'❌ Error: {str(e)}')

    # Check ANOVA results
    anova_path = tables_dir / 'anova_ttest_results.csv'
    if anova_path.exists():
        try:
            anova_df = pd.read_csv(anova_path)

            # Check all numerical variables tested
            if len(anova_df) >= 4:
                results.add_check('Content', 'ANOVA tests', True,
                                f'✅ {len(anova_df)} numerical variables tested')
            else:
                results.add_check('Content', 'ANOVA tests', False,
                                f'❌ Only {len(anova_df)} tests (expected 4+)')

            # Check effect sizes calculated
            if 'Cohens_D' in anova_df.columns:
                effect_sizes = anova_df['Cohens_D'].notna().all()
                if effect_sizes:
                    results.add_check('Content', 'Effect sizes', True, '✅ Cohen\'s D calculated')
                else:
                    results.add_check('Content', 'Effect sizes', False, '❌ Missing effect sizes')
        except Exception as e:
            results.add_check('Content', 'ANOVA analysis', False, f'❌ Error: {str(e)}')

    # Check correlation matrix
    corr_path = tables_dir / 'correlation_matrix.csv'
    if corr_path.exists():
        try:
            corr_df = pd.read_csv(corr_path, index_col=0)

            # Verify it's a square matrix
            if len(corr_df) == len(corr_df.columns):
                results.add_check('Content', 'Correlation matrix', True,
                                f'✅ {len(corr_df)}×{len(corr_df.columns)} matrix')
            else:
                results.add_check('Content', 'Correlation matrix', False,
                                '❌ Not a square matrix')

            # Check if Approved column exists
            if 'Approved' in corr_df.columns:
                results.add_check('Content', 'Approval correlations', True,
                                '✅ Approval correlations calculated')
            else:
                results.add_check('Content', 'Approval correlations', False,
                                '❌ Approval not in correlation matrix')
        except Exception as e:
            results.add_check('Content', 'Correlation matrix', False, f'❌ Error: {str(e)}')

    # ========================================================================
    # SUMMARY AND OUTPUT
    # ========================================================================
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    summary = results.get_summary()

    print(f"\n📊 Total Checks: {summary['total_checks']}")
    print(f"✅ Passed: {summary['passed']}")
    print(f"❌ Failed: {summary['failed']}")
    print(f"⚠️  Warnings: {summary['warnings']}")
    print(f"📈 Pass Rate: {summary['pass_rate']:.1%}")

    # Group checks by category
    print("\n" + "=" * 70)
    print("DETAILED RESULTS BY CATEGORY")
    print("=" * 70)

    categories = {}
    for check in results.checks:
        cat = check['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(check)

    for category, checks in categories.items():
        print(f"\n📁 {category.upper()}")
        print("-" * 70)

        for check in checks:
            status = "✅" if check['passed'] else "❌"
            print(f"  {status} {check['name']}: {check['message']}")

    # Generate insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)

    insights = []

    # Check for engineered features
    if features_path.exists():
        df = pd.read_csv(features_path)
        if 'DTI' in df.columns and 'LTI' in df.columns:
            insights.append(f"✅ Successfully engineered {len(['DTI', 'LTI', 'FICO_Bin_Custom', 'Income_Quartile', 'Loan_Category'])} new features")

    # Check statistical tests
    if chi_path.exists():
        chi_df = pd.read_csv(chi_path)
        sig_count = (chi_df['Significant'] == 'Yes').sum() if 'Significant' in chi_df.columns else 0
        insights.append(f"✅ {sig_count} categorical variables significantly predict approval")

    if anova_path.exists():
        anova_df = pd.read_csv(anova_path)
        insights.append(f"✅ All {len(anova_df)} numerical variables tested for significance")

    for insight in insights:
        print(f"  {insight}")

    # Output JSON for programmatic parsing
    validation_output = {
        'success': summary['pass_rate'] >= 0.80,
        'subphase': 'Phase 1.2: Bivariate Analysis',
        'checks_passed': summary['passed'],
        'total_checks': summary['total_checks'],
        'pass_rate': summary['pass_rate'],
        'details': [
            {
                'category': cat,
                'checks': [{'name': c['name'], 'passed': c['passed'], 'message': c['message']}
                          for c in checks]
            }
            for cat, checks in categories.items()
        ],
        'insights': insights
    }

    print("\n" + "=" * 70)
    print("__JSON_OUTPUT__")
    print(json.dumps(validation_output, indent=2))
    print("=" * 70)

    return validation_output['success']


if __name__ == "__main__":
    success = validate_phase1_bivariate()
    exit(0 if success else 1)
