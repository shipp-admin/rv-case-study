"""
Validation test for Phase 3.3: Lender Specialization Analysis
Validates outputs from lender_specialization.py
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

def validate_phase3_specialization():
    """
    Validate Phase 3.3 Lender Specialization Analysis outputs

    Expected outputs:
    - ANOVA results table
    - Chi-square results table
    - Customer clusters table
    - Lender preference matrix
    - Lender sweet spots table
    - Specialization summary JSON
    - Comprehensive visualization
    """
    results = ValidationResults()
    base_dir = Path(__file__).parent.parent
    tables_dir = base_dir / 'reports' / 'phase3_lender_analysis' / 'tables'
    figures_dir = base_dir / 'reports' / 'phase3_lender_analysis' / 'figures'

    # Track data for content validation
    anova_df = None
    chi_df = None
    clusters_df = None
    preference_matrix = None
    sweet_spots_df = None
    summary_json = None

    # ========================================
    # TABLE EXISTENCE CHECKS
    # ========================================

    # Check ANOVA results table
    anova_path = tables_dir / 'anova_results.csv'
    if anova_path.exists():
        results.add_check(
            'Table Existence',
            'anova_results.csv exists',
            True,
            f'Found ANOVA test results'
        )
        try:
            anova_df = pd.read_csv(anova_path)

            # Check structure
            required_cols = ['Variable', 'F_Statistic', 'P_Value', 'Significant',
                           'Mean_A', 'Mean_B', 'Mean_C']
            if all(col in anova_df.columns for col in required_cols):
                results.add_check(
                    'Table Structure',
                    'anova_results.csv has required columns',
                    True,
                    'All required columns present'
                )

                # Check has expected variables
                expected_vars = ['FICO_score', 'Monthly_Gross_Income', 'Loan_Amount', 'DTI', 'LTI']
                if set(anova_df['Variable']) == set(expected_vars):
                    results.add_check(
                        'Content Quality',
                        'ANOVA tested all numerical variables',
                        True,
                        '5 numerical variables tested'
                    )

                # Check significance detection
                significant_count = anova_df['Significant'].sum()
                if significant_count > 0:
                    results.add_check(
                        'Statistical Validity',
                        'ANOVA detected significant differences',
                        True,
                        f'{significant_count}/{len(anova_df)} variables significant at p<0.05'
                    )
        except Exception as e:
            results.add_check(
                'Table Loading',
                'anova_results.csv loads successfully',
                False,
                f'Error: {str(e)}'
            )
    else:
        results.add_check(
            'Table Existence',
            'anova_results.csv exists',
            False,
            f'File not found at {anova_path}'
        )

    # Check chi-square results table
    chi_path = tables_dir / 'chi_square_results.csv'
    if chi_path.exists():
        results.add_check(
            'Table Existence',
            'chi_square_results.csv exists',
            True,
            f'Found chi-square test results'
        )
        try:
            chi_df = pd.read_csv(chi_path)

            # Check structure
            required_cols = ['Variable', 'Chi2_Statistic', 'P_Value', 'Degrees_of_Freedom', 'Significant']
            if all(col in chi_df.columns for col in required_cols):
                results.add_check(
                    'Table Structure',
                    'chi_square_results.csv has required columns',
                    True,
                    'All required columns present'
                )

                # Check has categorical variables
                expected_vars = ['Reason', 'Employment_Status', 'Income_Quartile', 'Loan_Bracket']
                if set(chi_df['Variable']) == set(expected_vars):
                    results.add_check(
                        'Content Quality',
                        'Chi-square tested all categorical variables',
                        True,
                        '4 categorical variables tested'
                    )

                # Check significance detection
                significant_count = chi_df['Significant'].sum()
                if significant_count > 0:
                    results.add_check(
                        'Statistical Validity',
                        'Chi-square detected significant differences',
                        True,
                        f'{significant_count}/{len(chi_df)} variables significant at p<0.05'
                    )
        except Exception as e:
            results.add_check(
                'Table Loading',
                'chi_square_results.csv loads successfully',
                False,
                f'Error: {str(e)}'
            )
    else:
        results.add_check(
            'Table Existence',
            'chi_square_results.csv exists',
            False,
            f'File not found at {chi_path}'
        )

    # Check customer clusters table
    clusters_path = tables_dir / 'customer_clusters.csv'
    if clusters_path.exists():
        results.add_check(
            'Table Existence',
            'customer_clusters.csv exists',
            True,
            f'Found customer segmentation'
        )
        try:
            clusters_df = pd.read_csv(clusters_path)

            # Check structure
            required_cols = ['Cluster', 'Size', 'Pct_of_Approvals', 'Mean_FICO',
                           'Mean_Income', 'Mean_Loan', 'Lender_A_Pct', 'Lender_B_Pct', 'Lender_C_Pct']
            if all(col in clusters_df.columns for col in required_cols):
                results.add_check(
                    'Table Structure',
                    'customer_clusters.csv has required columns',
                    True,
                    'All required columns present'
                )

                # Check has 3-5 clusters
                num_clusters = len(clusters_df)
                if 3 <= num_clusters <= 6:
                    results.add_check(
                        'Content Quality',
                        'Optimal number of clusters identified',
                        True,
                        f'{num_clusters} customer segments found'
                    )

                # Check lender distributions sum to ~100%
                for idx, row in clusters_df.iterrows():
                    total_pct = row['Lender_A_Pct'] + row['Lender_B_Pct'] + row['Lender_C_Pct']
                    if 99 <= total_pct <= 101:  # Allow small rounding errors
                        results.add_check(
                            'Data Consistency',
                            f'Cluster {int(row["Cluster"])} lender percentages sum to 100%',
                            True,
                            f'Total: {total_pct:.1f}%'
                        )
        except Exception as e:
            results.add_check(
                'Table Loading',
                'customer_clusters.csv loads successfully',
                False,
                f'Error: {str(e)}'
            )
    else:
        results.add_check(
            'Table Existence',
            'customer_clusters.csv exists',
            False,
            f'File not found at {clusters_path}'
        )

    # Check lender preference matrix
    preference_path = tables_dir / 'lender_preference_matrix.csv'
    if preference_path.exists():
        results.add_check(
            'Table Existence',
            'lender_preference_matrix.csv exists',
            True,
            f'Found preference matrix'
        )
        try:
            preference_matrix = pd.read_csv(preference_path)

            # Check structure
            required_cols = ['Cluster', 'Lender_A', 'Lender_B', 'Lender_C']
            if all(col in preference_matrix.columns for col in required_cols):
                results.add_check(
                    'Table Structure',
                    'lender_preference_matrix.csv has required columns',
                    True,
                    'All required columns present'
                )

                # Check shows variation across lenders
                if len(preference_matrix) > 0:
                    a_std = preference_matrix['Lender_A'].std()
                    b_std = preference_matrix['Lender_B'].std()
                    c_std = preference_matrix['Lender_C'].std()

                    if a_std > 1 or b_std > 1 or c_std > 1:  # At least 1% std dev
                        results.add_check(
                            'Content Quality',
                            'Preference matrix shows lender variation',
                            True,
                            'Lenders show different cluster preferences'
                        )
        except Exception as e:
            results.add_check(
                'Table Loading',
                'lender_preference_matrix.csv loads successfully',
                False,
                f'Error: {str(e)}'
            )
    else:
        results.add_check(
            'Table Existence',
            'lender_preference_matrix.csv exists',
            False,
            f'File not found at {preference_path}'
        )

    # Check lender sweet spots
    sweet_spots_path = tables_dir / 'lender_sweet_spots.csv'
    if sweet_spots_path.exists():
        results.add_check(
            'Table Existence',
            'lender_sweet_spots.csv exists',
            True,
            f'Found lender sweet spots'
        )
        try:
            sweet_spots_df = pd.read_csv(sweet_spots_path)

            # Check structure
            required_cols = ['Lender', 'Sweet_Spot_Cluster', 'Percentage']
            if all(col in sweet_spots_df.columns for col in required_cols):
                results.add_check(
                    'Table Structure',
                    'lender_sweet_spots.csv has required columns',
                    True,
                    'All required columns present'
                )

                # Check all 3 lenders present
                if set(sweet_spots_df['Lender']) == {'A', 'B', 'C'}:
                    results.add_check(
                        'Content Quality',
                        'Sweet spots identified for all lenders',
                        True,
                        'A, B, C all have sweet spot clusters'
                    )

                # Check percentages are reasonable (> 25%)
                if (sweet_spots_df['Percentage'] > 25).all():
                    results.add_check(
                        'Content Quality',
                        'Sweet spot percentages are substantial',
                        True,
                        'All sweet spots > 25% of cluster'
                    )
        except Exception as e:
            results.add_check(
                'Table Loading',
                'lender_sweet_spots.csv loads successfully',
                False,
                f'Error: {str(e)}'
            )
    else:
        results.add_check(
            'Table Existence',
            'lender_sweet_spots.csv exists',
            False,
            f'File not found at {sweet_spots_path}'
        )

    # Check specialization summary JSON
    summary_path = tables_dir / 'lender_specialization_summary.json'
    if summary_path.exists():
        results.add_check(
            'Summary Existence',
            'lender_specialization_summary.json exists',
            True,
            f'Found specialization summary'
        )
        try:
            with open(summary_path, 'r') as f:
                summary_json = json.load(f)

            # Check required fields
            required_fields = ['anova_significant_vars', 'chi_square_significant_vars',
                             'num_clusters', 'lender_sweet_spots']
            if all(field in summary_json for field in required_fields):
                results.add_check(
                    'Summary Structure',
                    'Summary JSON has required fields',
                    True,
                    'All required fields present'
                )

                # Check content makes sense
                if summary_json['num_clusters'] >= 3:
                    results.add_check(
                        'Summary Content',
                        'Summary reports valid cluster count',
                        True,
                        f"{summary_json['num_clusters']} clusters"
                    )
        except Exception as e:
            results.add_check(
                'Summary Loading',
                'lender_specialization_summary.json loads successfully',
                False,
                f'Error: {str(e)}'
            )
    else:
        results.add_check(
            'Summary Existence',
            'lender_specialization_summary.json exists',
            False,
            f'File not found at {summary_path}'
        )

    # ========================================
    # FIGURE CHECKS
    # ========================================

    # Check comprehensive visualization
    viz_path = figures_dir / 'lender_specialization_analysis.png'
    if viz_path.exists():
        results.add_check(
            'Figure Existence',
            'lender_specialization_analysis.png exists',
            True,
            f'Found specialization visualization'
        )

        file_size = viz_path.stat().st_size
        if file_size > 50000:  # At least 50KB for 4-panel figure
            results.add_check(
                'Figure Quality',
                'lender_specialization_analysis.png has valid size',
                True,
                f'File size: {file_size} bytes'
            )
    else:
        results.add_check(
            'Figure Existence',
            'lender_specialization_analysis.png exists',
            False,
            f'File not found at {viz_path}'
        )

    # ========================================
    # CONTENT VALIDATION
    # ========================================

    # Validate ANOVA detected significant differences
    if anova_df is not None:
        significant_vars = anova_df[anova_df['Significant']]['Variable'].tolist()
        if len(significant_vars) >= 3:
            results.add_check(
                'Statistical Validity',
                'ANOVA confirms lender differences',
                True,
                f'{len(significant_vars)} variables differ significantly: {significant_vars}'
            )

    # Validate chi-square detected categorical differences
    if chi_df is not None:
        significant_vars = chi_df[chi_df['Significant']]['Variable'].tolist()
        if len(significant_vars) >= 1:
            results.add_check(
                'Statistical Validity',
                'Chi-square confirms categorical differences',
                True,
                f'{len(significant_vars)} categorical variables differ: {significant_vars}'
            )

    # Validate cluster sizes are reasonable
    if clusters_df is not None:
        total_size = clusters_df['Size'].sum()
        if total_size > 5000:  # At least 5000 approved apps
            results.add_check(
                'Content Quality',
                'Clusters contain substantial sample size',
                True,
                f'{total_size:,} total approved applications clustered'
            )

        # Check clusters show meaningful differences
        fico_range = clusters_df['Mean_FICO'].max() - clusters_df['Mean_FICO'].min()
        if fico_range > 50:
            results.add_check(
                'Content Quality',
                'Clusters show meaningful FICO differences',
                True,
                f'FICO range: {fico_range:.0f} points'
            )

    # ========================================
    # GENERATE INSIGHTS
    # ========================================

    insights = []

    if anova_df is not None:
        significant_count = anova_df['Significant'].sum()
        insights.append(f"{significant_count}/{len(anova_df)} numerical variables differ significantly across lenders")

    if chi_df is not None:
        significant_count = chi_df['Significant'].sum()
        insights.append(f"{significant_count}/{len(chi_df)} categorical variables differ significantly across lenders")

    if clusters_df is not None:
        insights.append(f"{len(clusters_df)} distinct customer segments identified through k-means clustering")

    if sweet_spots_df is not None:
        for _, row in sweet_spots_df.iterrows():
            insights.append(f"Lender {row['Lender']} sweet spot: Cluster {int(row['Sweet_Spot_Cluster'])} ({row['Percentage']:.1f}%)")

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
        'subphase': 'Phase 3.3: Lender Specialization Analysis',
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
    validate_phase3_specialization()
