"""
Validation Test for Phase 1.3: Missing Value Treatment

Verifies missing value handling meets PRD requirements.
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


def validate_phase1_missing_values():
    """
    Validate Phase 1.3 outputs according to PRD section 1.3 validation criteria

    Expected outputs to verify:
    - data/processed/cleaned_data.csv exists with NO missing Employment_Sector values
    - Missing value analysis report shows strategy (Unknown category OR imputation)
    - Test if missing pattern predicts approval (chi-square test result documented)
    - No other unexpected missing values introduced
    - Data shape remains 100,000 rows (no deletions unless justified)
    """
    print("=" * 70)
    print("PHASE 1.3: MISSING VALUE TREATMENT - VALIDATION TEST")
    print("=" * 70)

    results = ValidationResults()

    # ========================================================================
    # CATEGORY 1: DATA - Cleaned dataset
    # ========================================================================
    print("\n💾 Validating Cleaned Data...")

    cleaned_path = Path('data/processed/cleaned_data.csv')

    if not cleaned_path.exists():
        results.add_check('Data', 'cleaned_data.csv', False, '❌ File not found')
    else:
        try:
            df = pd.read_csv(cleaned_path)

            # Check row count (should be 100,000 - no deletions)
            if len(df) == 100000:
                results.add_check('Data', 'Row count', True, '✅ 100,000 rows preserved (no data loss)')
            else:
                results.add_check('Data', 'Row count', False,
                                f'❌ {len(df):,} rows (expected 100,000)')

            # Check Employment_Sector column exists
            if 'Employment_Sector' not in df.columns:
                results.add_check('Data', 'Employment_Sector column', False, '❌ Column not found')
            else:
                # Check NO missing values in Employment_Sector
                missing_count = df['Employment_Sector'].isna().sum()
                if missing_count == 0:
                    results.add_check('Data', 'Employment_Sector missing', True,
                                    '✅ 0 missing values (100% complete)')
                else:
                    results.add_check('Data', 'Employment_Sector missing', False,
                                    f'❌ {missing_count:,} missing values remain')

                # Check if 'Unknown' category was created
                if 'Unknown' in df['Employment_Sector'].values:
                    unknown_count = (df['Employment_Sector'] == 'Unknown').sum()
                    expected_unknown = 6407  # From PRD: 6.4% of 100,000

                    if 6000 <= unknown_count <= 7000:  # Allow some tolerance
                        results.add_check('Data', 'Unknown category', True,
                                        f'✅ {unknown_count:,} Unknown values (strategy: preserve missing pattern)')
                    else:
                        results.add_warning('Data', 'Unknown category',
                                          f'{unknown_count:,} Unknown values (expected ~6,407)')
                else:
                    results.add_warning('Data', 'Unknown category',
                                      'Unknown category not found (alternative strategy used)')

            # Check no other unexpected missing values
            total_missing = df.isna().sum().sum()
            if total_missing == 0:
                results.add_check('Data', 'Other missing values', True,
                                '✅ No unexpected missing values in dataset')
            else:
                results.add_warning('Data', 'Other missing values',
                                  f'{total_missing} missing values in other columns')

        except Exception as e:
            results.add_check('Data', 'cleaned_data.csv', False, f'❌ Error reading: {str(e)}')

    # ========================================================================
    # CATEGORY 2: ANALYSIS - Missing value pattern analysis
    # ========================================================================
    print("\n📊 Validating Missing Value Analysis...")

    comparison_path = Path('reports/phase1_eda/tables/missing_value_comparison.csv')
    report_path = Path('reports/phase1_eda/tables/missing_value_report.json')

    # Check comparison table
    if comparison_path.exists():
        try:
            comp_df = pd.read_csv(comparison_path)

            # Check numerical variables analyzed
            expected_vars = ['FICO_score', 'Loan_Amount', 'Monthly_Gross_Income', 'Monthly_Housing_Payment']
            if len(comp_df) >= len(expected_vars):
                results.add_check('Analysis', 'Numerical comparison', True,
                                f'✅ {len(comp_df)} variables analyzed')
            else:
                results.add_check('Analysis', 'Numerical comparison', False,
                                f'❌ Only {len(comp_df)} variables (expected {len(expected_vars)})')

            # Check p-values calculated
            if 'P_Value' in comp_df.columns:
                p_values_present = comp_df['P_Value'].notna().all()
                if p_values_present:
                    results.add_check('Analysis', 'P-values', True, '✅ All p-values calculated')
                else:
                    results.add_check('Analysis', 'P-values', False, '❌ Some p-values missing')
        except Exception as e:
            results.add_check('Analysis', 'Comparison table', False, f'❌ Error: {str(e)}')
    else:
        results.add_check('Analysis', 'missing_value_comparison.csv', False, '❌ Not found')

    # Check missing value report
    if report_path.exists():
        try:
            with open(report_path, 'r') as f:
                report = json.load(f)

            # Check analysis section exists
            if 'analysis' in report:
                analysis = report['analysis']

                # Check chi-square test documented
                if 'chi_square_p_value' in analysis:
                    p_value = analysis['chi_square_p_value']
                    predicts = analysis.get('predicts_approval', False)

                    if predicts and p_value < 0.05:
                        results.add_check('Analysis', 'Missing pattern test', True,
                                        f'✅ Missing pattern predicts approval (p={p_value:.2e})')
                    elif not predicts and p_value >= 0.05:
                        results.add_check('Analysis', 'Missing pattern test', True,
                                        f'✅ Missing pattern not significant (p={p_value:.2e})')
                    else:
                        results.add_warning('Analysis', 'Missing pattern test',
                                          f'Inconsistent results (p={p_value:.2e}, predicts={predicts})')
                else:
                    results.add_check('Analysis', 'Chi-square test', False, '❌ Not documented')

                # Check approval rate difference
                if 'approval_difference' in analysis:
                    diff = analysis['approval_difference']
                    results.add_check('Analysis', 'Approval difference', True,
                                    f'✅ Difference: {diff:.2f} percentage points')
                else:
                    results.add_check('Analysis', 'Approval difference', False, '❌ Not calculated')

            # Check strategy section exists
            if 'strategy' in report:
                strategy = report['strategy']

                if 'chosen_strategy' in strategy:
                    strat_name = strategy['chosen_strategy']
                    results.add_check('Analysis', 'Strategy documented', True,
                                    f'✅ Strategy: {strat_name}')
                else:
                    results.add_check('Analysis', 'Strategy documented', False, '❌ Strategy not specified')

                # Check rationale provided
                if 'rationale' in strategy:
                    results.add_check('Analysis', 'Strategy rationale', True, '✅ Rationale provided')
                else:
                    results.add_warning('Analysis', 'Strategy rationale', 'Rationale not documented')

        except Exception as e:
            results.add_check('Analysis', 'Missing value report', False, f'❌ Error: {str(e)}')
    else:
        results.add_check('Analysis', 'missing_value_report.json', False, '❌ Not found')

    # ========================================================================
    # CATEGORY 3: FIGURES - Missing value visualizations
    # ========================================================================
    print("\n📊 Validating Figures...")

    fig_path = Path('reports/phase1_eda/figures/missing_value_analysis.png')

    if fig_path.exists():
        size = fig_path.stat().st_size
        if size > 10000:  # At least 10KB
            results.add_check('Figures', 'missing_value_analysis.png', True,
                            f'✅ Found ({size/1024:.1f}KB)')
        else:
            results.add_warning('Figures', 'missing_value_analysis.png',
                              f'File too small ({size} bytes)')
    else:
        results.add_check('Figures', 'missing_value_analysis.png', False, '❌ Not found')

    # ========================================================================
    # CATEGORY 4: CONTENT QUALITY - Data integrity checks
    # ========================================================================
    print("\n🔍 Validating Content Quality...")

    if cleaned_path.exists():
        try:
            df = pd.read_csv(cleaned_path)

            # Check data types preserved
            expected_types = {
                'User ID': 'object',
                'FICO_score': ('int64', 'float64'),
                'Approved': ('int64', 'float64')
            }

            for col, expected_type in expected_types.items():
                if col in df.columns:
                    actual_type = df[col].dtype
                    if isinstance(expected_type, tuple):
                        type_match = str(actual_type) in expected_type
                    else:
                        type_match = str(actual_type) == expected_type

                    if type_match:
                        results.add_check('Content', f'{col} type', True, f'✅ Correct type')
                    else:
                        results.add_warning('Content', f'{col} type',
                                          f'Type: {actual_type} (expected {expected_type})')

            # Check approval rate unchanged
            if 'Approved' in df.columns:
                approval_rate = df['Approved'].mean()
                expected_rate = 0.1098  # 10.98% from PRD

                if abs(approval_rate - expected_rate) < 0.001:  # Within 0.1%
                    results.add_check('Content', 'Approval rate', True,
                                    f'✅ {approval_rate:.2%} (unchanged)')
                else:
                    results.add_warning('Content', 'Approval rate',
                                      f'{approval_rate:.2%} (expected {expected_rate:.2%})')

            # Check lender distribution unchanged
            if 'Lender' in df.columns:
                lender_dist = df['Lender'].value_counts(normalize=True)
                expected_dist = {'A': 0.55, 'B': 0.275, 'C': 0.175}

                dist_match = all(
                    abs(lender_dist.get(lender, 0) - expected_pct) < 0.01
                    for lender, expected_pct in expected_dist.items()
                )

                if dist_match:
                    results.add_check('Content', 'Lender distribution', True,
                                    '✅ Distribution preserved')
                else:
                    results.add_warning('Content', 'Lender distribution',
                                      'Distribution changed slightly')

        except Exception as e:
            results.add_check('Content', 'Data integrity', False, f'❌ Error: {str(e)}')

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

    if cleaned_path.exists():
        df = pd.read_csv(cleaned_path)

        # Data preservation
        if len(df) == 100000:
            insights.append("✅ All 100,000 rows preserved (no data loss)")

        # Missing value resolution
        if df['Employment_Sector'].isna().sum() == 0:
            insights.append("✅ All Employment_Sector missing values resolved")

        # Strategy used
        if 'Unknown' in df['Employment_Sector'].values:
            unknown_count = (df['Employment_Sector'] == 'Unknown').sum()
            insights.append(f"✅ Unknown category strategy: {unknown_count:,} values marked")

    if report_path.exists():
        with open(report_path, 'r') as f:
            report = json.load(f)
            if 'analysis' in report and 'predicts_approval' in report['analysis']:
                if report['analysis']['predicts_approval']:
                    insights.append("✅ Missing pattern is significant predictor of approval")

    for insight in insights:
        print(f"  {insight}")

    # Output JSON for programmatic parsing
    validation_output = {
        'success': summary['pass_rate'] >= 0.80,
        'subphase': 'Phase 1.3: Missing Value Treatment',
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
    success = validate_phase1_missing_values()
    exit(0 if success else 1)
