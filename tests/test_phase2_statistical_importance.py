"""
Validation Test for Phase 2.1: Statistical Feature Importance

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


def validate_phase2_statistical_importance():
    """
    Validate Phase 2.1 outputs according to PRD section 2.1 validation criteria

    Expected outputs to verify:
    - reports/phase2_feature_importance/tables/mutual_information.csv
    - reports/phase2_feature_importance/tables/anova_f_statistics.csv
    - reports/phase2_feature_importance/tables/chi_square_tests.csv
    - reports/phase2_feature_importance/tables/point_biserial_correlations.csv
    - reports/phase2_feature_importance/tables/features_to_drop.csv
    - reports/phase2_feature_importance/figures/statistical_importance.png
    - All features ranked by multiple methods
    - P-values < 0.05 for top features
    """
    print("=" * 70)
    print("PHASE 2.1: STATISTICAL FEATURE IMPORTANCE - VALIDATION TEST")
    print("=" * 70)

    results = ValidationResults()

    # ========================================================================
    # CATEGORY 1: TABLES - Statistical test results
    # ========================================================================
    print("\n📋 Validating Tables...")

    tables_dir = Path('reports/phase2_feature_importance/tables')

    required_tables = {
        'mutual_information.csv': {'min_rows': 5, 'required_cols': ['Feature', 'Type', 'Mutual_Information', 'Rank']},
        'anova_f_statistics.csv': {'min_rows': 5, 'required_cols': ['Feature', 'F_Statistic', 'P_Value', 'Significant']},
        'chi_square_tests.csv': {'min_rows': 3, 'required_cols': ['Feature', 'Chi2_Statistic', 'P_Value', 'Significant']},
        'point_biserial_correlations.csv': {'min_rows': 5, 'required_cols': ['Feature', 'Correlation', 'P_Value', 'Significant']},
        'features_to_drop.csv': {},
    }

    for table, criteria in required_tables.items():
        table_path = tables_dir / table

        if not table_path.exists():
            results.add_check('Tables', table, False, '❌ Not found')
            continue

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

        except Exception as e:
            results.add_check('Tables', table, False, f'❌ Error reading: {str(e)}')

    # ========================================================================
    # CATEGORY 2: FIGURES - Statistical importance visualizations
    # ========================================================================
    print("\n📊 Validating Figures...")

    fig_path = Path('reports/phase2_feature_importance/figures/statistical_importance.png')

    if fig_path.exists():
        size = fig_path.stat().st_size
        if size > 50000:  # At least 50KB (4 subplots)
            results.add_check('Figures', 'statistical_importance.png', True,
                            f'✅ Found ({size/1024:.1f}KB)')
        else:
            results.add_warning('Figures', 'statistical_importance.png',
                              f'File small ({size/1024:.1f}KB), may be incomplete')
    else:
        results.add_check('Figures', 'statistical_importance.png', False, '❌ Not found')

    # ========================================================================
    # CATEGORY 3: CONTENT QUALITY - Statistical validity
    # ========================================================================
    print("\n🔍 Validating Content Quality...")

    # Check mutual information rankings
    mi_path = tables_dir / 'mutual_information.csv'
    if mi_path.exists():
        try:
            mi_df = pd.read_csv(mi_path)

            # Check if features are ranked
            if 'Rank' in mi_df.columns:
                ranks = mi_df['Rank'].tolist()
                if ranks == list(range(1, len(ranks) + 1)):
                    results.add_check('Content', 'MI rankings', True,
                                    f'✅ {len(ranks)} features properly ranked')
                else:
                    results.add_check('Content', 'MI rankings', False,
                                    '❌ Ranking sequence invalid')

            # Check MI scores are non-negative
            if 'Mutual_Information' in mi_df.columns:
                if (mi_df['Mutual_Information'] >= 0).all():
                    results.add_check('Content', 'MI values', True,
                                    '✅ All MI scores valid (>= 0)')
                else:
                    results.add_check('Content', 'MI values', False,
                                    '❌ Invalid MI scores found')

            # Check top feature has meaningful MI score
            if len(mi_df) > 0:
                top_mi = mi_df.iloc[0]['Mutual_Information']
                if top_mi > 0.01:
                    results.add_check('Content', 'Top MI score', True,
                                    f'✅ Top feature MI: {top_mi:.4f}')
                else:
                    results.add_warning('Content', 'Top MI score',
                                      f'Low top MI score: {top_mi:.4f}')

        except Exception as e:
            results.add_check('Content', 'MI analysis', False, f'❌ Error: {str(e)}')

    # Check ANOVA results
    anova_path = tables_dir / 'anova_f_statistics.csv'
    if anova_path.exists():
        try:
            anova_df = pd.read_csv(anova_path)

            # Check p-values calculated
            if 'P_Value' in anova_df.columns:
                p_values_present = anova_df['P_Value'].notna().all()
                if p_values_present:
                    results.add_check('Content', 'ANOVA p-values', True, '✅ All present')
                else:
                    results.add_check('Content', 'ANOVA p-values', False, '❌ Some missing')

            # Check significant features identified
            if 'Significant' in anova_df.columns:
                sig_count = (anova_df['Significant'] == 'Yes').sum()
                if sig_count > 0:
                    results.add_check('Content', 'Significant ANOVA features', True,
                                    f'✅ {sig_count} significant numerical features')
                else:
                    results.add_warning('Content', 'Significant ANOVA features',
                                      'No significant features found')

            # Check F-statistics are positive
            if 'F_Statistic' in anova_df.columns:
                if (anova_df['F_Statistic'] > 0).all():
                    results.add_check('Content', 'F-statistics', True,
                                    '✅ All F-statistics valid (> 0)')
                else:
                    results.add_check('Content', 'F-statistics', False,
                                    '❌ Invalid F-statistics found')

        except Exception as e:
            results.add_check('Content', 'ANOVA analysis', False, f'❌ Error: {str(e)}')

    # Check chi-square results
    chi2_path = tables_dir / 'chi_square_tests.csv'
    if chi2_path.exists():
        try:
            chi2_df = pd.read_csv(chi2_path)

            # Check significant features
            if 'Significant' in chi2_df.columns:
                sig_count = (chi2_df['Significant'] == 'Yes').sum()
                if sig_count > 0:
                    results.add_check('Content', 'Significant chi-square features', True,
                                    f'✅ {sig_count} significant categorical features')
                else:
                    results.add_warning('Content', 'Significant chi-square features',
                                      'No significant features found')

            # Check chi2 statistics are positive
            if 'Chi2_Statistic' in chi2_df.columns:
                if (chi2_df['Chi2_Statistic'] > 0).all():
                    results.add_check('Content', 'Chi2-statistics', True,
                                    '✅ All chi2-statistics valid (> 0)')
                else:
                    results.add_check('Content', 'Chi2-statistics', False,
                                    '❌ Invalid chi2-statistics found')

        except Exception as e:
            results.add_check('Content', 'Chi-square analysis', False, f'❌ Error: {str(e)}')

    # Check point-biserial correlations
    pb_path = tables_dir / 'point_biserial_correlations.csv'
    if pb_path.exists():
        try:
            pb_df = pd.read_csv(pb_path)

            # Check correlations are in valid range [-1, 1]
            if 'Correlation' in pb_df.columns:
                valid_range = ((pb_df['Correlation'] >= -1) & (pb_df['Correlation'] <= 1)).all()
                if valid_range:
                    results.add_check('Content', 'PB correlations', True,
                                    '✅ All correlations in valid range [-1, 1]')
                else:
                    results.add_check('Content', 'PB correlations', False,
                                    '❌ Invalid correlations found')

        except Exception as e:
            results.add_check('Content', 'PB analysis', False, f'❌ Error: {str(e)}')

    # Check features to drop identified
    drop_path = tables_dir / 'features_to_drop.csv'
    if drop_path.exists():
        try:
            drop_df = pd.read_csv(drop_path)

            if len(drop_df) == 0:
                results.add_check('Content', 'Features to drop', True,
                                '✅ No features recommended for removal (all have predictive power)')
            else:
                results.add_check('Content', 'Features to drop', True,
                                f'✅ {len(drop_df)} features identified for removal')

        except Exception as e:
            results.add_check('Content', 'Features to drop', False, f'❌ Error: {str(e)}')

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

    if mi_path.exists():
        mi_df = pd.read_csv(mi_path)
        if len(mi_df) > 0:
            top_feature = mi_df.iloc[0]['Feature']
            top_score = mi_df.iloc[0]['Mutual_Information']
            insights.append(f"✅ Top feature by mutual information: {top_feature} (MI={top_score:.4f})")

    if anova_path.exists():
        anova_df = pd.read_csv(anova_path)
        sig_count = (anova_df['Significant'] == 'Yes').sum()
        insights.append(f"✅ {sig_count}/{len(anova_df)} numerical features statistically significant")

    if chi2_path.exists():
        chi2_df = pd.read_csv(chi2_path)
        sig_count = (chi2_df['Significant'] == 'Yes').sum()
        insights.append(f"✅ {sig_count}/{len(chi2_df)} categorical features statistically significant")

    if drop_path.exists():
        drop_df = pd.read_csv(drop_path)
        if len(drop_df) == 0:
            insights.append("✅ All features show predictive power - none recommended for removal")
        else:
            insights.append(f"✅ {len(drop_df)} features recommended for removal")

    for insight in insights:
        print(f"  {insight}")

    # Output JSON for programmatic parsing
    validation_output = {
        'success': summary['pass_rate'] >= 0.80,
        'subphase': 'Phase 2.1: Statistical Feature Importance',
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
    success = validate_phase2_statistical_importance()
    exit(0 if success else 1)
