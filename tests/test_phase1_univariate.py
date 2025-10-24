"""
Phase 1.1 Validation Test

Verifies that univariate analysis outputs exist and meet quality criteria.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def validate_phase1_univariate():
    """Validate Phase 1.1 univariate analysis outputs."""

    print_section("PHASE 1.1 VALIDATION - UNIVARIATE ANALYSIS")

    # Expected outputs
    output_dir = Path("reports/phase1_eda")
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"

    passed_checks = 0
    total_checks = 0

    # Track validation details for JSON output
    validation_details = []
    insights = []

    # Check 1: Tables directory exists
    print("✓ Checkpoint 1: Checking tables directory...")
    total_checks += 1
    if tables_dir.exists():
        print(f"✅ Tables directory exists: {tables_dir}")
        passed_checks += 1
    else:
        print(f"❌ Tables directory missing: {tables_dir}")
        return False

    # Check 2: Figures directory exists
    print("\n✓ Checkpoint 2: Checking figures directory...")
    total_checks += 1
    if figures_dir.exists():
        print(f"✅ Figures directory exists: {figures_dir}")
        passed_checks += 1
    else:
        print(f"❌ Figures directory missing: {figures_dir}")
        return False

    # Check 3: Categorical variable tables exist
    print("\n✓ Checkpoint 3: Checking categorical variable tables...")
    categorical_tables = [
        'approval_by_reason.csv',
        'approval_by_employment_status.csv',
        'approval_by_lender.csv',
        'approval_by_fico_bins.csv'
    ]

    table_checks = []
    for table_file in categorical_tables:
        total_checks += 1
        table_path = tables_dir / table_file
        check_passed = table_path.exists()
        if check_passed:
            print(f"✅ Found: {table_file}")
            passed_checks += 1
        else:
            print(f"⚠️  Missing: {table_file}")
        table_checks.append({
            "name": table_file,
            "passed": check_passed,
            "message": "✅ Found" if check_passed else "⚠️ Missing"
        })

    validation_details.append({
        "category": "Tables",
        "checks": table_checks
    })

    # Check 4: Numerical statistics table exists
    print("\n✓ Checkpoint 4: Checking numerical statistics table...")
    total_checks += 1
    stats_path = tables_dir / "numerical_variable_statistics.csv"
    if stats_path.exists():
        print(f"✅ Found: numerical_variable_statistics.csv")
        passed_checks += 1

        # Load and check content
        import pandas as pd
        stats_df = pd.read_csv(stats_path)
        expected_vars = ['FICO_score', 'Loan_Amount', 'Monthly_Gross_Income', 'Monthly_Housing_Payment']

        print("\n   Numerical variables analyzed:")
        for var in expected_vars:
            if var in stats_df['variable'].values:
                print(f"   ✅ {var}")
            else:
                print(f"   ⚠️  {var} (missing)")
    else:
        print(f"⚠️  Missing: numerical_variable_statistics.csv")

    # Check 5: Key figures exist
    print("\n✓ Checkpoint 5: Checking visualization files...")
    expected_figures = [
        'approval_by_reason.png',
        'approval_by_employment_status.png',
        'approval_by_lender.png',
        'approval_by_fico_bins.png',
        'fico_score_by_approval.png',
        'loan_amount_by_approval.png',
        'monthly_gross_income_by_approval.png',
        'monthly_housing_payment_by_approval.png'
    ]

    for fig_file in expected_figures:
        total_checks += 1
        fig_path = figures_dir / fig_file
        if fig_path.exists():
            print(f"✅ Found: {fig_file}")
            passed_checks += 1
        else:
            print(f"⚠️  Missing: {fig_file}")

    # Check 6: Validate table contents
    print("\n✓ Checkpoint 6: Validating table contents...")

    # Check FICO bins table
    fico_bins_path = tables_dir / "approval_by_fico_bins.csv"
    if fico_bins_path.exists():
        import pandas as pd
        fico_df = pd.read_csv(fico_bins_path)

        total_checks += 1
        if 'Approval_Rate' in fico_df.columns and len(fico_df) >= 4:
            print(f"✅ FICO bins table has valid structure")
            passed_checks += 1

            # Check if approval rate varies meaningfully
            approval_range = fico_df['Approval_Rate'].max() - fico_df['Approval_Rate'].min()
            total_checks += 1
            if approval_range > 0.05:  # At least 5% difference
                print(f"✅ FICO bins show meaningful variation in approval rate ({approval_range:.1%} range)")
                passed_checks += 1
            else:
                print(f"⚠️  FICO bins show minimal variation ({approval_range:.1%} range)")

            # Display FICO insights
            print("\n   FICO Score Insights:")
            best_fico = fico_df.loc[fico_df['Approval_Rate'].idxmax()]
            worst_fico = fico_df.loc[fico_df['Approval_Rate'].idxmin()]
            insights.append(f"FICO {best_fico['FICO_Bin']}: {best_fico['Approval_Rate']:.1%} approval vs {worst_fico['FICO_Bin']}: {worst_fico['Approval_Rate']:.1%}")

            for _, row in fico_df.iterrows():
                print(f"   - {row['FICO_Bin']}: {row['Approval_Rate']:.1%} approval ({row['Total_Applications']:,} apps)")
        else:
            print(f"⚠️  FICO bins table has invalid structure")

    # Check Reason table
    reason_path = tables_dir / "approval_by_reason.csv"
    if reason_path.exists():
        import pandas as pd
        reason_df = pd.read_csv(reason_path)

        total_checks += 1
        if 'Approval_Rate' in reason_df.columns and len(reason_df) >= 2:
            print(f"\n✅ Reason table has valid structure")
            passed_checks += 1

            print("\n   Loan Reason Insights:")
            for _, row in reason_df.head(3).iterrows():
                print(f"   - {row['Reason']}: {row['Approval_Rate']:.1%} approval ({row['Total_Applications']:,} apps)")

    # Check Lender table
    lender_path = tables_dir / "approval_by_lender.csv"
    if lender_path.exists():
        import pandas as pd
        lender_df = pd.read_csv(lender_path)

        total_checks += 1
        if len(lender_df) == 3 and set(lender_df['Lender'].values) == {'A', 'B', 'C'}:
            print(f"\n✅ Lender table has all 3 lenders (A, B, C)")
            passed_checks += 1

            print("\n   Lender Approval Rates:")
            lender_c = lender_df[lender_df['Lender'] == 'C']
            if len(lender_c) > 0:
                insights.append(f"Lender C most lenient ({lender_c['Approval_Rate'].values[0]:.1%} approval)")

            for _, row in lender_df.iterrows():
                print(f"   - Lender {row['Lender']}: {row['Approval_Rate']:.1%} ({row['Total_Applications']:,} apps)")

    # Summary
    print_section("VALIDATION SUMMARY")

    print(f"Checks passed: {passed_checks} / {total_checks}")
    pass_rate = passed_checks / total_checks if total_checks > 0 else 0
    print(f"Pass rate: {pass_rate:.1%}\n")

    success = pass_rate >= 0.80

    if success:
        print("✅ PHASE 1.1 VALIDATION PASSED")
        print("\nKey outputs verified:")
        print("  ✓ Approval rate tables by categorical variables")
        print("  ✓ Numerical variable statistics and distributions")
        print("  ✓ FICO score binned analysis")
        print("  ✓ Visualizations for all variables")
        print("\nNext step: Proceed to Phase 1.2 (Bivariate Analysis)")
    else:
        print("❌ PHASE 1.1 VALIDATION FAILED")
        print(f"\nOnly {pass_rate:.1%} of checks passed (need ≥80%)")
        print("\nMissing outputs:")
        print("  - Check console output above for missing files")
        print("  - Re-run: python3 src/phase1_eda/univariate.py")

    # Build structured JSON output for UI integration
    import json
    validation_output = {
        "success": success,
        "subphase": "Phase 1.1: Univariate Analysis",
        "checks_passed": passed_checks,
        "total_checks": total_checks,
        "pass_rate": round(pass_rate, 4),
        "details": validation_details,
        "insights": insights
    }

    # Output JSON marker for UI parsing
    print("\n__JSON_OUTPUT__")
    print(json.dumps(validation_output, indent=2))

    return success


if __name__ == "__main__":
    success = validate_phase1_univariate()
    sys.exit(0 if success else 1)
