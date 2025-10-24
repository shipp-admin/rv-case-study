"""
Validation test for Phase 3.1: Lender Approval Profiling
Validates outputs from lender_profiling.py
"""

import os
import json
import pandas as pd
from pathlib import Path

class ValidationResults:
    def __init__(self):
        self.checks = []
        self.passed = 0
        self.failed = 0

    def add_check(self, category, name, passed, message=""):
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

    def get_summary(self):
        total = self.passed + self.failed
        return {
            'total_checks': total,
            'passed': self.passed,
            'failed': self.failed,
            'pass_rate': self.passed / total if total > 0 else 0
        }

def validate_phase3_lender_profiling():
    """
    Validate Phase 3.1 Lender Approval Profiling outputs

    Expected outputs:
    - Overall lender approval rates table
    - 3 lender profile reports (A, B, C) with segment breakdowns
    - 3 lender heatmaps
    - 1 cross-lender comparison chart
    """
    results = ValidationResults()
    base_dir = Path(__file__).parent.parent
    tables_dir = base_dir / 'reports' / 'phase3_lender_analysis' / 'tables'
    figures_dir = base_dir / 'reports' / 'phase3_lender_analysis' / 'figures'

    lenders = ['A', 'B', 'C']

    # Track data for content validation
    approval_rates_df = None
    lender_comparisons = {}
    lender_thresholds = {}

    # ========================================
    # TABLE EXISTENCE CHECKS
    # ========================================

    # Check overall approval rates table
    rates_path = tables_dir / 'lender_approval_rates.csv'
    if rates_path.exists():
        results.add_check(
            'Table Existence',
            'lender_approval_rates.csv exists',
            True,
            f'Found at {rates_path}'
        )
        try:
            approval_rates_df = pd.read_csv(rates_path)

            # Check structure
            required_cols = ['Lender', 'Total_Applications', 'Approved', 'Approval_Rate']
            if all(col in approval_rates_df.columns for col in required_cols):
                results.add_check(
                    'Table Structure',
                    'lender_approval_rates.csv has required columns',
                    True,
                    f'All columns present'
                )
            else:
                results.add_check(
                    'Table Structure',
                    'lender_approval_rates.csv has required columns',
                    False,
                    f'Missing columns'
                )

            # Check has all lenders
            if set(approval_rates_df['Lender']) == set(lenders):
                results.add_check(
                    'Content Quality',
                    'All 3 lenders present in approval rates',
                    True,
                    'A, B, C all found'
                )
            else:
                results.add_check(
                    'Content Quality',
                    'All 3 lenders present in approval rates',
                    False,
                    f"Found: {approval_rates_df['Lender'].tolist()}"
                )
        except Exception as e:
            results.add_check(
                'Table Structure',
                'lender_approval_rates.csv is valid',
                False,
                f'Error: {str(e)}'
            )
    else:
        results.add_check(
            'Table Existence',
            'lender_approval_rates.csv exists',
            False,
            f'File not found at {rates_path}'
        )

    # Check lender-specific tables
    for lender in lenders:
        # Segment tables (at least FICO groups)
        fico_path = tables_dir / f'lender_{lender}_fico_groups.csv'
        if fico_path.exists():
            results.add_check(
                'Table Existence',
                f'lender_{lender}_fico_groups.csv exists',
                True,
                f'Found segment analysis for Lender {lender}'
            )
        else:
            results.add_check(
                'Table Existence',
                f'lender_{lender}_fico_groups.csv exists',
                False,
                f'Missing segment analysis for Lender {lender}'
            )

        # Statistical comparisons
        comp_path = tables_dir / f'lender_{lender}_comparisons.csv'
        if comp_path.exists():
            results.add_check(
                'Table Existence',
                f'lender_{lender}_comparisons.csv exists',
                True,
                f'Found comparisons for Lender {lender}'
            )
            try:
                comp_df = pd.read_csv(comp_path)
                lender_comparisons[lender] = comp_df

                # Check for significant differences
                if 'P_Value' in comp_df.columns and 'Significant' in comp_df.columns:
                    significant_count = comp_df['Significant'].sum()
                    results.add_check(
                        'Content Quality',
                        f'Lender {lender} has significant differences',
                        True,
                        f'{significant_count}/{len(comp_df)} metrics significantly different'
                    )
            except Exception as e:
                results.add_check(
                    'Content Quality',
                    f'Lender {lender} comparisons are valid',
                    False,
                    f'Error: {str(e)}'
                )
        else:
            results.add_check(
                'Table Existence',
                f'lender_{lender}_comparisons.csv exists',
                False,
                f'Missing comparisons for Lender {lender}'
            )

        # Thresholds
        thresh_path = tables_dir / f'lender_{lender}_thresholds.csv'
        if thresh_path.exists():
            results.add_check(
                'Table Existence',
                f'lender_{lender}_thresholds.csv exists',
                True,
                f'Found thresholds for Lender {lender}'
            )
            try:
                thresh_df = pd.read_csv(thresh_path)
                lender_thresholds[lender] = thresh_df

                # Check threshold values are reasonable
                required_thresholds = ['min_fico', 'min_income', 'max_lti', 'bankruptcy_approval_rate']
                if all(col in thresh_df.columns for col in required_thresholds):
                    min_fico = thresh_df['min_fico'].iloc[0]
                    if 300 <= min_fico <= 850:
                        results.add_check(
                            'Content Quality',
                            f'Lender {lender} min FICO is reasonable',
                            True,
                            f'Min FICO: {min_fico:.0f}'
                        )
                    else:
                        results.add_check(
                            'Content Quality',
                            f'Lender {lender} min FICO is reasonable',
                            False,
                            f'Min FICO out of range: {min_fico}'
                        )
            except Exception as e:
                results.add_check(
                    'Content Quality',
                    f'Lender {lender} thresholds are valid',
                    False,
                    f'Error: {str(e)}'
                )
        else:
            results.add_check(
                'Table Existence',
                f'lender_{lender}_thresholds.csv exists',
                False,
                f'Missing thresholds for Lender {lender}'
            )

    # ========================================
    # FIGURE EXISTENCE CHECKS
    # ========================================

    # Check lender heatmaps
    for lender in lenders:
        heatmap_path = figures_dir / f'lender_{lender}_heatmap.png'
        if heatmap_path.exists():
            results.add_check(
                'Figure Existence',
                f'lender_{lender}_heatmap.png exists',
                True,
                f'Found heatmap for Lender {lender}'
            )

            # Check file size
            file_size = heatmap_path.stat().st_size
            if file_size > 10000:
                results.add_check(
                    'Figure Quality',
                    f'lender_{lender}_heatmap.png has valid size',
                    True,
                    f'File size: {file_size} bytes'
                )
            else:
                results.add_check(
                    'Figure Quality',
                    f'lender_{lender}_heatmap.png has valid size',
                    False,
                    f'File size too small: {file_size} bytes'
                )
        else:
            results.add_check(
                'Figure Existence',
                f'lender_{lender}_heatmap.png exists',
                False,
                f'Missing heatmap for Lender {lender}'
            )

    # Check comparison chart
    comparison_path = figures_dir / 'lender_comparison.png'
    if comparison_path.exists():
        results.add_check(
            'Figure Existence',
            'lender_comparison.png exists',
            True,
            f'Found cross-lender comparison'
        )

        file_size = comparison_path.stat().st_size
        if file_size > 10000:
            results.add_check(
                'Figure Quality',
                'lender_comparison.png has valid size',
                True,
                f'File size: {file_size} bytes'
            )
    else:
        results.add_check(
            'Figure Existence',
            'lender_comparison.png exists',
            False,
            f'Missing comparison chart'
        )

    # ========================================
    # CONTENT VALIDATION
    # ========================================

    # Validate approval rates
    if approval_rates_df is not None:
        # Check approval rates are within reasonable range
        if (approval_rates_df['Approval_Rate'] >= 0).all() and (approval_rates_df['Approval_Rate'] <= 1).all():
            results.add_check(
                'Content Quality',
                'Approval rates are valid percentages',
                True,
                f"Range: {approval_rates_df['Approval_Rate'].min():.2%} - {approval_rates_df['Approval_Rate'].max():.2%}"
            )

        # Check lenders have different approval rates (variability)
        rate_std = approval_rates_df['Approval_Rate'].std()
        if rate_std > 0.01:  # At least 1% standard deviation
            results.add_check(
                'Content Quality',
                'Lenders show variability in approval rates',
                True,
                f'Std dev: {rate_std:.4f}'
            )

    # Validate statistical tests show significance
    if lender_comparisons:
        p_values = []
        for lender, comp_df in lender_comparisons.items():
            if 'P_Value' in comp_df.columns:
                p_values.extend(comp_df['P_Value'].tolist())

        if p_values:
            significant_count = sum(1 for p in p_values if p < 0.01)
            if significant_count > 0:
                results.add_check(
                    'Content Quality',
                    'Statistical tests confirm lender differences',
                    True,
                    f'{significant_count}/{len(p_values)} tests significant at p<0.01'
                )

    # Validate thresholds differ between lenders
    if len(lender_thresholds) == 3:
        min_ficos = [lender_thresholds[l]['min_fico'].iloc[0] for l in lenders]
        fico_range = max(min_ficos) - min(min_ficos)

        if fico_range > 50:  # At least 50 point difference
            results.add_check(
                'Content Quality',
                'Lender FICO thresholds differ substantially',
                True,
                f'Range: {min(min_ficos):.0f} - {max(min_ficos):.0f} (spread: {fico_range:.0f})'
            )

    # ========================================
    # GENERATE INSIGHTS
    # ========================================

    insights = []

    if approval_rates_df is not None:
        for _, row in approval_rates_df.iterrows():
            insights.append(f"Lender {row['Lender']}: {row['Approval_Rate']:.2%} approval rate")

    if lender_thresholds:
        min_fico_summary = ', '.join([f"{l}: {lender_thresholds[l]['min_fico'].iloc[0]:.0f}"
                                      for l in lenders if l in lender_thresholds])
        insights.append(f"Min FICO thresholds - {min_fico_summary}")

    if lender_comparisons:
        total_significant = sum(
            comp_df['Significant'].sum()
            for comp_df in lender_comparisons.values()
            if 'Significant' in comp_df.columns
        )
        insights.append(f"{total_significant} statistically significant differences detected across lenders")

    # ========================================
    # COMPILE RESULTS
    # ========================================

    summary = results.get_summary()

    # Group checks by category
    details = []
    categories = {}
    for check in results.checks:
        cat = check['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(check)

    for category, checks in categories.items():
        details.append({
            'category': category,
            'checks': checks
        })

    validation_output = {
        'success': summary['pass_rate'] >= 0.80,
        'subphase': 'Phase 3.1: Lender Approval Profiling',
        'checks_passed': summary['passed'],
        'total_checks': summary['total_checks'],
        'pass_rate': summary['pass_rate'],
        'details': details,
        'insights': insights
    }

    # Output JSON for dashboard consumption
    print("__JSON_OUTPUT__")
    print(json.dumps(validation_output, indent=2))
    print("__JSON_OUTPUT_END__")

    return validation_output

if __name__ == "__main__":
    validate_phase3_lender_profiling()
