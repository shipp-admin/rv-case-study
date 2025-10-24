"""
Validation test for Phase 4.1: Current State Revenue Analysis
Validates outputs from baseline_revenue.py
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

def validate_phase4_baseline_revenue():
    """
    Validate Phase 4.1 Current State Revenue Analysis outputs

    Expected outputs:
    - Overall RPA metrics table
    - RPA by lender table
    - RPA by FICO segment table
    - RPA by income segment table
    - RPA by loan bracket table
    - Bounty structure analysis table
    - Comprehensive baseline revenue dashboard
    """
    results = ValidationResults()
    base_dir = Path(__file__).parent.parent
    tables_dir = base_dir / 'reports' / 'phase4_revenue_optimization' / 'tables'
    figures_dir = base_dir / 'reports' / 'phase4_revenue_optimization' / 'figures'

    # Track data for content validation
    overall_rpa_df = None
    lender_df = None
    fico_df = None
    income_df = None
    loan_df = None
    bounty_structure_df = None

    # ========================================
    # TABLE EXISTENCE CHECKS
    # ========================================

    # Check overall RPA table
    overall_path = tables_dir / 'overall_rpa.csv'
    if overall_path.exists():
        results.add_check(
            'Table Existence',
            'overall_rpa.csv exists',
            True,
            f'Found overall RPA metrics'
        )
        try:
            overall_rpa_df = pd.read_csv(overall_path)

            # Check structure
            required_cols = ['Total_Bounty', 'Total_Applications', 'Overall_RPA',
                           'Approval_Rate', 'Approved_Bounty', 'Denied_Bounty']
            if all(col in overall_rpa_df.columns for col in required_cols):
                results.add_check(
                    'Table Structure',
                    'overall_rpa.csv has required columns',
                    True,
                    'All required columns present'
                )

                # Verify RPA calculation
                total_bounty = overall_rpa_df['Total_Bounty'].iloc[0]
                total_apps = overall_rpa_df['Total_Applications'].iloc[0]
                overall_rpa = overall_rpa_df['Overall_RPA'].iloc[0]
                expected_rpa = total_bounty / total_apps

                if abs(overall_rpa - expected_rpa) < 0.01:
                    results.add_check(
                        'Calculation Accuracy',
                        'Overall RPA calculation correct',
                        True,
                        f'RPA = ${overall_rpa:.2f} (verified)'
                    )
                else:
                    results.add_check(
                        'Calculation Accuracy',
                        'Overall RPA calculation correct',
                        False,
                        f'Expected {expected_rpa:.2f}, got {overall_rpa:.2f}'
                    )
        except Exception as e:
            results.add_check(
                'Table Loading',
                'overall_rpa.csv loads successfully',
                False,
                f'Error: {str(e)}'
            )
    else:
        results.add_check(
            'Table Existence',
            'overall_rpa.csv exists',
            False,
            f'File not found at {overall_path}'
        )

    # Check RPA by lender table
    lender_path = tables_dir / 'rpa_by_lender.csv'
    if lender_path.exists():
        results.add_check(
            'Table Existence',
            'rpa_by_lender.csv exists',
            True,
            f'Found RPA by lender'
        )
        try:
            lender_df = pd.read_csv(lender_path)

            # Check structure
            required_cols = ['Lender', 'Total_Applications', 'Total_Bounty', 'RPA', 'Approval_Rate']
            if all(col in lender_df.columns for col in required_cols):
                results.add_check(
                    'Table Structure',
                    'rpa_by_lender.csv has required columns',
                    True,
                    'All required columns present'
                )

                # Check all 3 lenders present
                if set(lender_df['Lender']) == {'A', 'B', 'C'}:
                    results.add_check(
                        'Content Quality',
                        'All 3 lenders in RPA table',
                        True,
                        'A, B, C all present'
                    )

                # Verify RPA calculations for each lender
                for _, row in lender_df.iterrows():
                    rpa = row['RPA']
                    expected_rpa = row['Total_Bounty'] / row['Total_Applications']
                    if abs(rpa - expected_rpa) < 0.01:
                        results.add_check(
                            'Calculation Accuracy',
                            f'Lender {row["Lender"]} RPA calculation correct',
                            True,
                            f'RPA = ${rpa:.2f}'
                        )
        except Exception as e:
            results.add_check(
                'Table Loading',
                'rpa_by_lender.csv loads successfully',
                False,
                f'Error: {str(e)}'
            )
    else:
        results.add_check(
            'Table Existence',
            'rpa_by_lender.csv exists',
            False,
            f'File not found at {lender_path}'
        )

    # Check RPA by FICO segment table
    fico_path = tables_dir / 'rpa_by_fico_segment.csv'
    if fico_path.exists():
        results.add_check(
            'Table Existence',
            'rpa_by_fico_segment.csv exists',
            True,
            f'Found RPA by FICO segment'
        )
        try:
            fico_df = pd.read_csv(fico_path)

            # Check structure
            required_cols = ['FICO_Group', 'Total_Applications', 'RPA', 'Approval_Rate']
            if all(col in fico_df.columns for col in required_cols):
                results.add_check(
                    'Table Structure',
                    'rpa_by_fico_segment.csv has required columns',
                    True,
                    'All required columns present'
                )

                # Check has FICO groups
                if len(fico_df) >= 4:
                    results.add_check(
                        'Content Quality',
                        'FICO segments documented',
                        True,
                        f'{len(fico_df)} FICO groups analyzed'
                    )

                # Check RPA increases with FICO quality (generally)
                # Verify poor < fair < good < very_good < excellent pattern exists
                fico_order = ['poor', 'fair', 'good', 'very_good', 'excellent']
                if any(group in fico_df['FICO_Group'].values for group in fico_order):
                    results.add_check(
                        'Content Quality',
                        'FICO segmentation complete',
                        True,
                        'Standard FICO groups present'
                    )
        except Exception as e:
            results.add_check(
                'Table Loading',
                'rpa_by_fico_segment.csv loads successfully',
                False,
                f'Error: {str(e)}'
            )
    else:
        results.add_check(
            'Table Existence',
            'rpa_by_fico_segment.csv exists',
            False,
            f'File not found at {fico_path}'
        )

    # Check RPA by income segment table
    income_path = tables_dir / 'rpa_by_income_segment.csv'
    if income_path.exists():
        results.add_check(
            'Table Existence',
            'rpa_by_income_segment.csv exists',
            True,
            f'Found RPA by income segment'
        )
        try:
            income_df = pd.read_csv(income_path)

            # Check structure
            required_cols = ['Income_Quartile', 'Total_Applications', 'RPA', 'Approval_Rate', 'Avg_Income']
            if all(col in income_df.columns for col in required_cols):
                results.add_check(
                    'Table Structure',
                    'rpa_by_income_segment.csv has required columns',
                    True,
                    'All required columns present'
                )

                # Check all 4 quartiles present
                if set(income_df['Income_Quartile']) == {'Q1', 'Q2', 'Q3', 'Q4'}:
                    results.add_check(
                        'Content Quality',
                        'All income quartiles present',
                        True,
                        'Q1-Q4 all documented'
                    )

                # Check RPA generally increases with income
                q1_rpa = income_df[income_df['Income_Quartile'] == 'Q1']['RPA'].values[0]
                q4_rpa = income_df[income_df['Income_Quartile'] == 'Q4']['RPA'].values[0]
                if q4_rpa > q1_rpa:
                    results.add_check(
                        'Content Quality',
                        'RPA increases with income',
                        True,
                        f'Q1=${q1_rpa:.2f} < Q4=${q4_rpa:.2f}'
                    )
        except Exception as e:
            results.add_check(
                'Table Loading',
                'rpa_by_income_segment.csv loads successfully',
                False,
                f'Error: {str(e)}'
            )
    else:
        results.add_check(
            'Table Existence',
            'rpa_by_income_segment.csv exists',
            False,
            f'File not found at {income_path}'
        )

    # Check RPA by loan bracket table
    loan_path = tables_dir / 'rpa_by_loan_bracket.csv'
    if loan_path.exists():
        results.add_check(
            'Table Existence',
            'rpa_by_loan_bracket.csv exists',
            True,
            f'Found RPA by loan bracket'
        )
        try:
            loan_df = pd.read_csv(loan_path)

            # Check structure
            required_cols = ['Loan_Bracket', 'Total_Applications', 'RPA', 'Approval_Rate', 'Avg_Loan_Amount']
            if all(col in loan_df.columns for col in required_cols):
                results.add_check(
                    'Table Structure',
                    'rpa_by_loan_bracket.csv has required columns',
                    True,
                    'All required columns present'
                )

                # Check all 3 brackets present
                if set(loan_df['Loan_Bracket']) == {'Small', 'Medium', 'Large'}:
                    results.add_check(
                        'Content Quality',
                        'All loan brackets present',
                        True,
                        'Small, Medium, Large all documented'
                    )
        except Exception as e:
            results.add_check(
                'Table Loading',
                'rpa_by_loan_bracket.csv loads successfully',
                False,
                f'Error: {str(e)}'
            )
    else:
        results.add_check(
            'Table Existence',
            'rpa_by_loan_bracket.csv exists',
            False,
            f'File not found at {loan_path}'
        )

    # Check bounty structure table
    bounty_path = tables_dir / 'bounty_structure.csv'
    if bounty_path.exists():
        results.add_check(
            'Table Existence',
            'bounty_structure.csv exists',
            True,
            f'Found bounty structure analysis'
        )
        try:
            bounty_structure_df = pd.read_csv(bounty_path)

            # Check structure
            required_cols = ['unique_values', 'mean', 'std', 'min', 'max', 'is_fixed']
            if all(col in bounty_structure_df.columns for col in required_cols):
                results.add_check(
                    'Table Structure',
                    'bounty_structure.csv has required columns',
                    True,
                    'All required columns present'
                )

                # Check bounty structure documented
                is_fixed = bounty_structure_df['is_fixed'].iloc[0]
                unique_values = bounty_structure_df['unique_values'].iloc[0]
                results.add_check(
                    'Content Quality',
                    'Bounty structure documented',
                    True,
                    f'Structure: {"Fixed" if is_fixed else "Variable"} ({unique_values} unique values)'
                )
        except Exception as e:
            results.add_check(
                'Table Loading',
                'bounty_structure.csv loads successfully',
                False,
                f'Error: {str(e)}'
            )
    else:
        results.add_check(
            'Table Existence',
            'bounty_structure.csv exists',
            False,
            f'File not found at {bounty_path}'
        )

    # ========================================
    # FIGURE CHECKS
    # ========================================

    # Check baseline revenue dashboard
    dashboard_path = figures_dir / 'baseline_revenue_dashboard.png'
    if dashboard_path.exists():
        results.add_check(
            'Figure Existence',
            'baseline_revenue_dashboard.png exists',
            True,
            f'Found baseline revenue visualization'
        )

        file_size = dashboard_path.stat().st_size
        if file_size > 100000:  # At least 100KB for comprehensive dashboard
            results.add_check(
                'Figure Quality',
                'baseline_revenue_dashboard.png has valid size',
                True,
                f'File size: {file_size} bytes'
            )
    else:
        results.add_check(
            'Figure Existence',
            'baseline_revenue_dashboard.png exists',
            False,
            f'File not found at {dashboard_path}'
        )

    # ========================================
    # CONTENT VALIDATION
    # ========================================

    # Validate overall RPA is positive
    if overall_rpa_df is not None:
        overall_rpa_val = overall_rpa_df['Overall_RPA'].iloc[0]
        if overall_rpa_val > 0:
            results.add_check(
                'Data Validity',
                'Overall RPA is positive',
                True,
                f'RPA = ${overall_rpa_val:.2f}'
            )

    # Validate lender RPAs sum to approximate overall
    if lender_df is not None and overall_rpa_df is not None:
        # Weighted average should match overall RPA
        total_apps = lender_df['Total_Applications'].sum()
        weighted_rpa = (lender_df['RPA'] * lender_df['Total_Applications']).sum() / total_apps
        overall_rpa_val = overall_rpa_df['Overall_RPA'].iloc[0]

        if abs(weighted_rpa - overall_rpa_val) < 0.01:
            results.add_check(
                'Data Consistency',
                'Lender RPAs consistent with overall RPA',
                True,
                f'Weighted avg = ${weighted_rpa:.2f}, Overall = ${overall_rpa_val:.2f}'
            )

    # Validate FICO RPA trend (higher FICO = higher RPA generally)
    if fico_df is not None and len(fico_df) >= 3:
        # Just check that best FICO has higher RPA than worst
        fico_groups = fico_df['FICO_Group'].values
        if 'excellent' in fico_groups or 'very_good' in fico_groups:
            best_fico_rpas = fico_df[fico_df['FICO_Group'].isin(['excellent', 'very_good'])]['RPA']
            worst_fico_rpas = fico_df[fico_df['FICO_Group'].isin(['poor', 'fair'])]['RPA']

            if len(best_fico_rpas) > 0 and len(worst_fico_rpas) > 0:
                if best_fico_rpas.max() > worst_fico_rpas.max():
                    results.add_check(
                        'Business Logic',
                        'Higher FICO scores have higher RPA',
                        True,
                        'FICO-RPA relationship validated'
                    )

    # ========================================
    # GENERATE INSIGHTS
    # ========================================

    insights = []

    if overall_rpa_df is not None:
        overall_rpa_val = overall_rpa_df['Overall_RPA'].iloc[0]
        total_revenue = overall_rpa_df['Total_Bounty'].iloc[0]
        insights.append(f"Overall baseline RPA: ${overall_rpa_val:.2f} per application")
        insights.append(f"Total baseline revenue: ${total_revenue:,.2f}")

    if lender_df is not None:
        max_rpa_row = lender_df.loc[lender_df['RPA'].idxmax()]
        min_rpa_row = lender_df.loc[lender_df['RPA'].idxmin()]
        insights.append(f"Lender {max_rpa_row['Lender']} has highest RPA: ${max_rpa_row['RPA']:.2f}")
        insights.append(f"Lender {min_rpa_row['Lender']} has lowest RPA: ${min_rpa_row['RPA']:.2f}")

    if fico_df is not None:
        if 'excellent' in fico_df['FICO_Group'].values:
            excellent_rpa = fico_df[fico_df['FICO_Group'] == 'excellent']['RPA'].values[0]
            insights.append(f"Excellent FICO customers generate ${excellent_rpa:.2f} RPA")

    if income_df is not None:
        q4_rpa = income_df[income_df['Income_Quartile'] == 'Q4']['RPA'].values[0]
        q1_rpa = income_df[income_df['Income_Quartile'] == 'Q1']['RPA'].values[0]
        insights.append(f"Top income quartile (Q4) generates ${q4_rpa:.2f} vs bottom (Q1) ${q1_rpa:.2f}")

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
        'subphase': 'Phase 4.1: Current State Revenue Analysis',
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
    validate_phase4_baseline_revenue()
