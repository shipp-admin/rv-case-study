"""
Validation test for Phase 4.2: Optimal Matching Algorithm
Validates outputs from matching_algorithm.py
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
            'passed': bool(passed),  # Convert to native Python bool
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
        print("PHASE 4.2 VALIDATION RESULTS")
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


def validate_phase4_2():
    """Main validation function"""
    results = ValidationResults()

    # Paths
    tables_dir = Path("reports/phase4_revenue_optimization/tables")
    figures_dir = Path("reports/phase4_revenue_optimization/figures")

    # Constants
    TOTAL_APPLICATIONS = 100000
    LENDERS = ['A', 'B', 'C']
    MIN_MEAN_EV = 40.0
    MAX_LATENCY_MS = 50
    MIN_CONFIDENCE = 0.0

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
        "optimal_lender_assignments.csv",
        "optimal_vs_current_comparison.csv",
        "matching_algorithm_performance.csv"
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
    required_figures = ["matching_algorithm_results.png"]
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
        assignments_df = pd.read_csv(tables_dir / "optimal_lender_assignments.csv")
        comparison_df = pd.read_csv(tables_dir / "optimal_vs_current_comparison.csv")
        performance_df = pd.read_csv(tables_dir / "matching_algorithm_performance.csv")
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

    # ===== 2. OPTIMAL ASSIGNMENTS VALIDATION =====
    results.add_check(
        "Optimal Assignments",
        "Row count matches total applications",
        len(assignments_df) == TOTAL_APPLICATIONS,
        f"Expected {TOTAL_APPLICATIONS}, got {len(assignments_df)}"
    )

    # Check required columns
    required_cols = ['Optimal_Lender', 'Optimal_EV', 'EV_Difference', 'Assignment_Confidence',
                     'EV_A', 'EV_B', 'EV_C']
    missing_cols = [col for col in required_cols if col not in assignments_df.columns]
    results.add_check(
        "Optimal Assignments",
        "All required columns present",
        len(missing_cols) == 0,
        f"Missing: {missing_cols}" if missing_cols else "All columns present"
    )

    # Check lender values
    valid_lenders = set(LENDERS)
    actual_lenders = set(assignments_df['Optimal_Lender'].unique())
    results.add_check(
        "Optimal Assignments",
        "Optimal lender values are valid",
        actual_lenders.issubset(valid_lenders),
        f"Expected {valid_lenders}, got {actual_lenders}"
    )

    # Check EV values are non-negative
    ev_cols = ['EV_A', 'EV_B', 'EV_C', 'Optimal_EV']
    ev_valid = all(assignments_df[col].min() >= 0 for col in ev_cols if col in assignments_df.columns)
    results.add_check(
        "Optimal Assignments",
        "Expected values are non-negative",
        ev_valid
    )

    # Check EV difference is non-negative
    if 'EV_Difference' in assignments_df.columns:
        results.add_check(
            "Optimal Assignments",
            "EV difference is non-negative",
            assignments_df['EV_Difference'].min() >= 0,
            f"Min: {assignments_df['EV_Difference'].min():.4f}"
        )

    # Check assignment confidence range
    if 'Assignment_Confidence' in assignments_df.columns:
        conf_min = assignments_df['Assignment_Confidence'].min()
        conf_max = assignments_df['Assignment_Confidence'].max()
        results.add_check(
            "Optimal Assignments",
            "Assignment confidence in [0,1]",
            conf_min >= 0.0 and conf_max <= 1.0,
            f"Range: [{conf_min:.4f}, {conf_max:.4f}]"
        )

    # Check optimal lender matches max EV
    sample = assignments_df.sample(min(1000, len(assignments_df)), random_state=42)
    matches = 0
    for idx, row in sample.iterrows():
        evs = {'A': row['EV_A'], 'B': row['EV_B'], 'C': row['EV_C']}
        max_lender = max(evs, key=evs.get)
        if row['Optimal_Lender'] == max_lender:
            matches += 1

    match_rate = matches / len(sample) * 100
    results.add_check(
        "Optimal Assignments",
        "Optimal lender matches max EV (sample)",
        match_rate >= 99.0,
        f"{matches}/{len(sample)} match ({match_rate:.1f}%)"
    )

    # Check mean optimal EV
    if 'Optimal_EV' in assignments_df.columns:
        mean_ev = assignments_df['Optimal_EV'].mean()
        results.add_check(
            "Optimal Assignments",
            f"Mean optimal EV >= ${MIN_MEAN_EV}",
            mean_ev >= MIN_MEAN_EV,
            f"Mean EV: ${mean_ev:.2f}"
        )

    # Check confidence distribution
    if 'Assignment_Confidence' in assignments_df.columns:
        mean_conf = assignments_df['Assignment_Confidence'].mean()
        results.add_check(
            "Optimal Assignments",
            f"Mean confidence >= {MIN_CONFIDENCE}",
            mean_conf >= MIN_CONFIDENCE,
            f"Mean: {mean_conf:.4f}"
        )

    # ===== 3. COMPARISON VALIDATION (aggregated summary) =====
    results.add_check(
        "Comparison Analysis",
        "Comparison summary exists",
        len(comparison_df) > 0,
        f"Got {len(comparison_df)} rows"
    )

    # Check required columns for aggregated format
    comp_required = ['Total_Applications', 'Should_Switch', 'Pct_Switch']
    comp_missing = [col for col in comp_required if col not in comparison_df.columns]
    results.add_check(
        "Comparison Analysis",
        "All required summary columns present",
        len(comp_missing) == 0,
        f"Missing: {comp_missing}" if comp_missing else "All columns present"
    )

    # Check total applications
    if 'Total_Applications' in comparison_df.columns:
        total = int(comparison_df['Total_Applications'].iloc[0])
        results.add_check(
            "Comparison Analysis",
            "Total applications matches expected",
            total == TOTAL_APPLICATIONS,
            f"Expected {TOTAL_APPLICATIONS}, got {total}"
        )

    # Check some should switch
    if 'Should_Switch' in comparison_df.columns:
        switch_count = int(comparison_df['Should_Switch'].iloc[0])
        results.add_check(
            "Comparison Analysis",
            "Some applications should switch",
            switch_count > 0,
            f"{switch_count:,} applications should switch"
        )

        # Check switch percentage
        if 'Pct_Switch' in comparison_df.columns:
            switch_pct = float(comparison_df['Pct_Switch'].iloc[0])
            results.add_check(
                "Comparison Analysis",
                "Switch percentage in reasonable range",
                0 <= switch_pct <= 100,
                f"{switch_pct:.1f}% should switch"
            )

    # ===== 4. PERFORMANCE METRICS =====
    results.add_check(
        "Performance Metrics",
        "Performance metrics exist",
        len(performance_df) > 0
    )

    # Check latency
    if 'Latency_Per_Customer_ms' in performance_df.columns:
        latency = performance_df['Latency_Per_Customer_ms'].iloc[0]
        results.add_check(
            "Performance Metrics",
            f"Latency < {MAX_LATENCY_MS}ms per customer",
            latency < MAX_LATENCY_MS,
            f"Actual: {latency:.2f}ms"
        )

    # Check mean optimal EV recorded
    if 'Mean_Optimal_EV' in performance_df.columns:
        results.add_check(
            "Performance Metrics",
            "Mean optimal EV is positive",
            performance_df['Mean_Optimal_EV'].iloc[0] > 0
        )

    # Check total assignments
    if 'Total_Assignments' in performance_df.columns:
        total = performance_df['Total_Assignments'].iloc[0]
        results.add_check(
            "Performance Metrics",
            "Total assignments matches expected",
            total == TOTAL_APPLICATIONS,
            f"Expected {TOTAL_APPLICATIONS}, got {total}"
        )

    # ===== 5. BUSINESS LOGIC =====
    # Check lender distribution changed (using assignments_df)
    if 'Lender' in assignments_df.columns and 'Optimal_Lender' in assignments_df.columns:
        current_dist = assignments_df['Lender'].value_counts(normalize=True)
        optimal_dist = assignments_df['Optimal_Lender'].value_counts(normalize=True)
        diff = (current_dist - optimal_dist).abs()

        results.add_check(
            "Business Logic",
            "Optimal distribution differs from current",
            diff.max() > 0.01,
            f"Max difference: {diff.max():.3f}"
        )

    # Check EV calculation (EV should be P(approval) × bounty, so EV/bounty = P(approval) which should be in [0,1])
    BOUNTY = 240.66
    sample = assignments_df.sample(min(100, len(assignments_df)), random_state=42)
    ev_calc_correct = True
    for idx, row in sample.iterrows():
        # Derive P(approval) from EV
        p_approval_a = row['EV_A'] / BOUNTY
        if p_approval_a < 0.0 or p_approval_a > 1.0:
            ev_calc_correct = False
            break

    results.add_check(
        "Business Logic",
        "EV implies valid P(approval) (sample)",
        ev_calc_correct
    )

    # Check high confidence assignments exist
    if 'Assignment_Confidence' in assignments_df.columns:
        high_conf = (assignments_df['Assignment_Confidence'] >= 0.5).sum()
        high_conf_pct = 100 * high_conf / len(assignments_df)
        results.add_check(
            "Business Logic",
            "High-confidence assignments > 10%",
            high_conf_pct > 10,
            f"{high_conf_pct:.1f}% high-confidence"
        )

    # Check distribution not extreme
    if 'Optimal_Lender' in assignments_df.columns:
        dist = assignments_df['Optimal_Lender'].value_counts(normalize=True)
        results.add_check(
            "Business Logic",
            "No lender has 0% assignments",
            dist.min() > 0.0,
            f"Min: {dist.min():.3f}"
        )

        results.add_check(
            "Business Logic",
            "No lender has 100% assignments",
            dist.max() < 1.0,
            f"Max: {dist.max():.3f}"
        )

    # ===== 6. DATA QUALITY =====
    # Check no missing values
    key_cols = ['Optimal_Lender', 'Optimal_EV', 'Assignment_Confidence']
    missing_counts = {col: assignments_df[col].isna().sum() for col in key_cols if col in assignments_df.columns}
    all_complete = all(count == 0 for count in missing_counts.values())
    results.add_check(
        "Data Quality",
        "No missing values in key columns",
        all_complete,
        f"Missing counts: {missing_counts}" if not all_complete else "All complete"
    )

    # Check EV values reasonable
    MAX_EV = 250.0
    ev_reasonable = all(
        assignments_df[col].max() <= MAX_EV
        for col in ['EV_A', 'EV_B', 'EV_C', 'Optimal_EV']
        if col in assignments_df.columns
    )
    results.add_check(
        "Data Quality",
        f"EV values <= ${MAX_EV}",
        ev_reasonable
    )

    return results


def main():
    """Run validation and output results"""
    results = validate_phase4_2()
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
        "subphase": "Phase 4.2: Optimal Matching Algorithm - Validation",
        "checks_passed": summary['passed'],
        "total_checks": summary['total_checks'],
        "pass_rate": summary['pass_rate'] / 100.0,  # Convert to 0-1 range
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
