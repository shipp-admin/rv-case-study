"""
Validation test for Phase 4.3: Incremental Revenue Calculation
Validates outputs from incremental_revenue.py
"""

import os
import json
import pandas as pd
import numpy as np
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
            'passed': bool(passed),
            'message': str(message) if message else ""
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def get_summary(self):
        total = self.passed + self.failed
        pass_rate = (self.passed / total * 100) if total > 0 else 0
        return {
            'total_checks': total,
            'passed': self.passed,
            'failed': self.failed,
            'pass_rate': pass_rate
        }

    def print_results(self):
        """Print results grouped by category"""
        categories = {}
        for check in self.checks:
            cat = check['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(check)

        print("\n" + "="*80)
        print("PHASE 4.3 VALIDATION RESULTS")
        print("="*80)

        for category, checks in categories.items():
            print(f"\n{category}:")
            for check in checks:
                status = "✓" if check['passed'] else "✗"
                print(f"  {status} {check['name']}")
                if check['message']:
                    print(f"     → {check['message']}")

        summary = self.get_summary()
        print("\n" + "="*80)
        print(f"SUMMARY: {summary['passed']}/{summary['total_checks']} checks passed " +
              f"({summary['pass_rate']:.1f}%)")
        print("="*80)


def validate_phase4_3():
    """Main validation function"""
    results = ValidationResults()

    # Paths
    tables_dir = Path("reports/phase4_revenue_optimization/tables")
    figures_dir = Path("reports/phase4_revenue_optimization/figures")

    # Constants
    MIN_LIFT_PERCENTAGE = 50.0  # Expect at least 50% revenue lift
    BOOTSTRAP_ITERATIONS = 1000
    TOTAL_APPLICATIONS = 100000

    # ===== 1. FILE EXISTENCE =====
    results.add_check(
        "File Existence",
        "Tables directory exists",
        tables_dir.exists(),
        f"Path: {tables_dir}"
    )

    results.add_check(
        "File Existence",
        "Figures directory exists",
        figures_dir.exists(),
        f"Path: {figures_dir}"
    )

    # Check tables
    required_tables = [
        "incremental_revenue_summary.csv",
        "incremental_revenue_by_segment.csv",
        "bootstrap_confidence_intervals.csv",
        "sensitivity_analysis.csv"
    ]

    for table in required_tables:
        path = tables_dir / table
        results.add_check(
            "File Existence",
            f"Table: {table}",
            path.exists(),
            f"Path: {path}"
        )

    # Check figures
    required_figures = ["incremental_revenue_analysis.png"]
    for figure in required_figures:
        path = figures_dir / figure
        results.add_check(
            "File Existence",
            f"Figure: {figure}",
            path.exists(),
            f"Path: {path}"
        )

    # Load data for further checks
    try:
        summary_df = pd.read_csv(tables_dir / "incremental_revenue_summary.csv")
        segments_df = pd.read_csv(tables_dir / "incremental_revenue_by_segment.csv")
        bootstrap_df = pd.read_csv(tables_dir / "bootstrap_confidence_intervals.csv")
        sensitivity_df = pd.read_csv(tables_dir / "sensitivity_analysis.csv")
    except Exception as e:
        results.add_check(
            "Data Loading",
            "Load CSV files",
            False,
            f"Error: {str(e)}"
        )
        results.print_results()
        return results

    results.add_check(
        "Data Loading",
        "Successfully loaded all CSV files",
        True
    )

    # ===== 2. SUMMARY VALIDATION =====
    results.add_check(
        "Summary Metrics",
        "Summary has one row",
        len(summary_df) == 1,
        f"Expected 1 row, got {len(summary_df)}"
    )

    # Check required columns
    required_cols = ['baseline_total_revenue', 'optimal_total_revenue', 'incremental_revenue',
                     'lift_percentage', 'baseline_rpa', 'optimal_rpa', 'ci_lower', 'ci_upper']
    missing_cols = [col for col in required_cols if col not in summary_df.columns]
    results.add_check(
        "Summary Metrics",
        "All required columns present",
        len(missing_cols) == 0,
        f"Missing: {missing_cols}" if missing_cols else "All columns present"
    )

    # Check optimal > baseline
    if 'baseline_total_revenue' in summary_df.columns and 'optimal_total_revenue' in summary_df.columns:
        baseline = float(summary_df['baseline_total_revenue'].iloc[0])
        optimal = float(summary_df['optimal_total_revenue'].iloc[0])
        results.add_check(
            "Summary Metrics",
            "Optimal revenue > baseline revenue",
            optimal > baseline,
            f"Baseline: ${baseline:,.0f}, Optimal: ${optimal:,.0f}"
        )

    # Check lift percentage
    if 'lift_percentage' in summary_df.columns:
        lift_pct = float(summary_df['lift_percentage'].iloc[0])
        results.add_check(
            "Summary Metrics",
            f"Lift percentage >= {MIN_LIFT_PERCENTAGE}%",
            lift_pct >= MIN_LIFT_PERCENTAGE,
            f"Actual lift: {lift_pct:.1f}%"
        )

    # Check incremental revenue matches calculation
    if all(col in summary_df.columns for col in ['baseline_total_revenue', 'optimal_total_revenue', 'incremental_revenue']):
        baseline = float(summary_df['baseline_total_revenue'].iloc[0])
        optimal = float(summary_df['optimal_total_revenue'].iloc[0])
        incremental = float(summary_df['incremental_revenue'].iloc[0])
        expected_incremental = optimal - baseline

        results.add_check(
            "Summary Metrics",
            "Incremental revenue = optimal - baseline",
            abs(incremental - expected_incremental) < 1.0,
            f"Calculated: ${incremental:,.0f}, Expected: ${expected_incremental:,.0f}"
        )

    # Check RPA lift
    if 'baseline_rpa' in summary_df.columns and 'optimal_rpa' in summary_df.columns:
        baseline_rpa = float(summary_df['baseline_rpa'].iloc[0])
        optimal_rpa = float(summary_df['optimal_rpa'].iloc[0])
        results.add_check(
            "Summary Metrics",
            "Optimal RPA > baseline RPA",
            optimal_rpa > baseline_rpa,
            f"Baseline: ${baseline_rpa:.2f}, Optimal: ${optimal_rpa:.2f}"
        )

    # ===== 3. SEGMENT ANALYSIS VALIDATION =====
    results.add_check(
        "Segment Analysis",
        "Segments data exists",
        len(segments_df) > 0,
        f"Found {len(segments_df)} segments"
    )

    # Check segment types
    if 'segment_type' in segments_df.columns:
        segment_types = segments_df['segment_type'].unique()
        expected_types = ['FICO Group', 'Income Quartile', 'Loan Bracket']
        types_present = all(t in segment_types for t in expected_types)
        results.add_check(
            "Segment Analysis",
            "All segment types present",
            types_present,
            f"Found: {list(segment_types)}"
        )

    # Check required columns
    seg_required = ['segment_type', 'segment_value', 'applications', 'baseline_revenue',
                    'optimal_revenue', 'incremental_revenue', 'lift_percentage']
    seg_missing = [col for col in seg_required if col not in segments_df.columns]
    results.add_check(
        "Segment Analysis",
        "All required columns present",
        len(seg_missing) == 0,
        f"Missing: {seg_missing}" if seg_missing else "All columns present"
    )

    # Check segment applications sum to total
    if 'applications' in segments_df.columns and 'segment_type' in segments_df.columns:
        # Get one segment type to avoid double counting
        fico_segments = segments_df[segments_df['segment_type'] == 'FICO Group']
        if len(fico_segments) > 0:
            total_apps = fico_segments['applications'].sum()
            results.add_check(
                "Segment Analysis",
                "Segment applications sum to total",
                total_apps == TOTAL_APPLICATIONS,
                f"Expected {TOTAL_APPLICATIONS:,}, got {total_apps:,}"
            )

    # Check some segments have positive lift
    if 'incremental_revenue' in segments_df.columns:
        positive_lift = (segments_df['incremental_revenue'] > 0).sum()
        results.add_check(
            "Segment Analysis",
            "Some segments have positive lift",
            positive_lift > 0,
            f"{positive_lift}/{len(segments_df)} segments with positive lift"
        )

    # ===== 4. BOOTSTRAP CONFIDENCE INTERVALS =====
    results.add_check(
        "Bootstrap Analysis",
        f"Bootstrap has {BOOTSTRAP_ITERATIONS} iterations",
        len(bootstrap_df) == BOOTSTRAP_ITERATIONS,
        f"Expected {BOOTSTRAP_ITERATIONS}, got {len(bootstrap_df)}"
    )

    # Check required columns
    bootstrap_required = ['iteration', 'incremental_revenue']
    bootstrap_missing = [col for col in bootstrap_required if col not in bootstrap_df.columns]
    results.add_check(
        "Bootstrap Analysis",
        "All required columns present",
        len(bootstrap_missing) == 0,
        f"Missing: {bootstrap_missing}" if bootstrap_missing else "All columns present"
    )

    # Check CI bounds are reasonable
    if 'ci_lower' in summary_df.columns and 'ci_upper' in summary_df.columns:
        ci_lower = float(summary_df['ci_lower'].iloc[0])
        ci_upper = float(summary_df['ci_upper'].iloc[0])

        results.add_check(
            "Bootstrap Analysis",
            "CI upper > CI lower",
            ci_upper > ci_lower,
            f"CI: [${ci_lower:,.0f}, ${ci_upper:,.0f}]"
        )

        # Check CI width is reasonable (not too wide)
        ci_width = ci_upper - ci_lower
        mean_incremental = (ci_lower + ci_upper) / 2
        ci_width_pct = (ci_width / mean_incremental) * 100 if mean_incremental > 0 else 0

        results.add_check(
            "Bootstrap Analysis",
            "CI width is reasonable (<20% of mean)",
            ci_width_pct < 20.0,
            f"CI width: {ci_width_pct:.1f}% of mean"
        )

    # Check bootstrap values distribution
    if 'incremental_revenue' in bootstrap_df.columns:
        bootstrap_mean = bootstrap_df['incremental_revenue'].mean()
        bootstrap_std = bootstrap_df['incremental_revenue'].std()

        results.add_check(
            "Bootstrap Analysis",
            "Bootstrap mean is positive",
            bootstrap_mean > 0,
            f"Mean: ${bootstrap_mean:,.0f}"
        )

        results.add_check(
            "Bootstrap Analysis",
            "Bootstrap std dev is reasonable",
            bootstrap_std < bootstrap_mean * 0.1,  # Std dev < 10% of mean
            f"Std dev: ${bootstrap_std:,.0f} ({(bootstrap_std/bootstrap_mean)*100:.1f}% of mean)"
        )

    # ===== 5. SENSITIVITY ANALYSIS =====
    results.add_check(
        "Sensitivity Analysis",
        "Sensitivity data exists",
        len(sensitivity_df) > 0,
        f"Found {len(sensitivity_df)} scenarios"
    )

    # Check required columns
    sens_required = ['bounty_variation', 'adjusted_bounty', 'baseline_revenue',
                     'optimal_revenue', 'incremental_revenue', 'lift_percentage']
    sens_missing = [col for col in sens_required if col not in sensitivity_df.columns]
    results.add_check(
        "Sensitivity Analysis",
        "All required columns present",
        len(sens_missing) == 0,
        f"Missing: {sens_missing}" if sens_missing else "All columns present"
    )

    # Check multiple scenarios
    if len(sensitivity_df) > 0:
        results.add_check(
            "Sensitivity Analysis",
            "Multiple scenarios analyzed",
            len(sensitivity_df) >= 5,
            f"Found {len(sensitivity_df)} scenarios"
        )

    # Check baseline scenario (0% variation) exists
    if 'bounty_variation' in sensitivity_df.columns:
        has_baseline = (sensitivity_df['bounty_variation'] == '+0%').any()
        results.add_check(
            "Sensitivity Analysis",
            "Baseline scenario (+0%) present",
            has_baseline
        )

    # Check lift percentage consistent across variations
    if 'lift_percentage' in sensitivity_df.columns:
        lift_percentages = sensitivity_df['lift_percentage'].unique()
        # All should be approximately the same (lift % doesn't change with bounty scaling)
        lift_std = sensitivity_df['lift_percentage'].std()
        results.add_check(
            "Sensitivity Analysis",
            "Lift percentage consistent across variations",
            lift_std < 0.1,  # Very small variation expected
            f"Std dev: {lift_std:.4f}"
        )

    # ===== 6. BUSINESS LOGIC =====
    # Check incremental revenue scales with bounty
    if all(col in sensitivity_df.columns for col in ['adjusted_bounty', 'incremental_revenue']):
        # Sort by bounty
        sorted_df = sensitivity_df.sort_values('adjusted_bounty')
        incremental_values = sorted_df['incremental_revenue'].values

        # Check if incremental revenue increases with bounty
        is_increasing = all(incremental_values[i] <= incremental_values[i+1]
                           for i in range(len(incremental_values)-1))

        results.add_check(
            "Business Logic",
            "Incremental revenue increases with bounty",
            is_increasing,
            f"Values: ${incremental_values[0]:,.0f} to ${incremental_values[-1]:,.0f}"
        )

    # Check top segments make sense
    if all(col in segments_df.columns for col in ['incremental_revenue', 'lift_percentage']):
        top_segment = segments_df.nlargest(1, 'incremental_revenue').iloc[0]

        results.add_check(
            "Business Logic",
            "Top segment has positive lift",
            top_segment['incremental_revenue'] > 0,
            f"{top_segment['segment_type']}: {top_segment['segment_value']}"
        )

    # ===== 7. DATA QUALITY =====
    # Check no missing values in summary
    if len(summary_df) > 0:
        missing_summary = summary_df.isnull().sum().sum()
        results.add_check(
            "Data Quality",
            "No missing values in summary",
            missing_summary == 0,
            f"Found {missing_summary} missing values" if missing_summary > 0 else "All complete"
        )

    # Check no negative revenues
    if 'incremental_revenue' in segments_df.columns:
        # Some segments can have negative lift, that's okay
        results.add_check(
            "Data Quality",
            "Segment data has no null values",
            segments_df['incremental_revenue'].notna().all()
        )

    # Check bootstrap has no extreme outliers
    if 'incremental_revenue' in bootstrap_df.columns:
        q1 = bootstrap_df['incremental_revenue'].quantile(0.25)
        q3 = bootstrap_df['incremental_revenue'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr

        outliers = ((bootstrap_df['incremental_revenue'] < lower_bound) |
                   (bootstrap_df['incremental_revenue'] > upper_bound)).sum()
        outlier_pct = (outliers / len(bootstrap_df)) * 100

        results.add_check(
            "Data Quality",
            "Bootstrap has few extreme outliers (<1%)",
            outlier_pct < 1.0,
            f"Outliers: {outliers}/{len(bootstrap_df)} ({outlier_pct:.2f}%)"
        )

    return results


def main():
    """Run validation and output results"""
    results = validate_phase4_3()
    results.print_results()

    # Group checks by category for dashboard
    categories = {}
    for check in results.checks:
        cat = check['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(check)

    # Format details as grouped categories
    details = [
        {
            'category': category,
            'checks': checks
        }
        for category, checks in categories.items()
    ]

    # Output JSON for dashboard
    summary = results.get_summary()
    json_output = {
        "success": summary['pass_rate'] >= 80.0,
        "subphase": "Phase 4.3: Incremental Revenue Calculation - Validation",
        "checks_passed": summary['passed'],
        "total_checks": summary['total_checks'],
        "pass_rate": summary['pass_rate'] / 100.0,
        "details": details,
        "summary": summary,
        "checks": results.checks
    }

    print("\n__JSON_OUTPUT__")
    print(json.dumps(json_output, indent=2))
    print("__JSON_OUTPUT_END__")

    return 0 if summary['pass_rate'] >= 80.0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
