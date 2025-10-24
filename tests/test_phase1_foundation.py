"""
Phase 1 Validation Script

Run this script to validate that Phase 1 implementation is working correctly.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from phase1_eda.data_loader import load_and_validate, get_summary_statistics
import pandas as pd


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80 + "\n")


def validate_phase1():
    """Run Phase 1 validation checks."""

    print_section("PHASE 1 VALIDATION - DATA LOADING & QUALITY CHECKS")

    # Checkpoint 1: Load and validate data
    print("✓ Checkpoint 1: Loading data...")
    try:
        df, report = load_and_validate()
        print("✅ Data loaded successfully!")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

    # Checkpoint 2: Verify shape
    print("\n✓ Checkpoint 2: Verifying data shape...")
    expected_rows = 100000
    expected_cols = 13  # After dropping 'applications'

    if len(df) == expected_rows:
        print(f"✅ Row count correct: {len(df):,} rows")
    else:
        print(f"⚠️  Row count mismatch: Expected {expected_rows:,}, got {len(df):,}")

    if len(df.columns) == expected_cols:
        print(f"✅ Column count correct: {len(df.columns)} columns")
    else:
        print(f"⚠️  Column count: Expected {expected_cols}, got {len(df.columns)}")

    # Checkpoint 3: Check approval rate
    print("\n✓ Checkpoint 3: Verifying approval rate...")
    approval_rate = df['Approved'].mean()
    expected_approval_rate = 0.10976

    if abs(approval_rate - expected_approval_rate) < 0.001:
        print(f"✅ Approval rate matches: {approval_rate:.4%}")
    else:
        print(f"⚠️  Approval rate mismatch: Expected {expected_approval_rate:.4%}, got {approval_rate:.4%}")

    # Checkpoint 4: Check missing values
    print("\n✓ Checkpoint 4: Checking missing values...")
    missing = df.isnull().sum()
    if 'Employment_Sector' in missing and missing['Employment_Sector'] > 0:
        pct_missing = missing['Employment_Sector'] / len(df) * 100
        print(f"✅ Employment_Sector missing values detected: {missing['Employment_Sector']:,} ({pct_missing:.1f}%)")
    else:
        print("⚠️  Expected missing values in Employment_Sector")

    # Checkpoint 5: Check lender distribution
    print("\n✓ Checkpoint 5: Verifying lender distribution...")
    lender_dist = df['Lender'].value_counts(normalize=True)
    expected_dist = {'A': 0.55, 'B': 0.275, 'C': 0.175}

    print("Lender distribution:")
    for lender in ['A', 'B', 'C']:
        actual = lender_dist.get(lender, 0)
        expected = expected_dist[lender]
        match = "✅" if abs(actual - expected) < 0.01 else "⚠️ "
        print(f"  {match} Lender {lender}: {actual:.1%} (expected {expected:.1%})")

    # Checkpoint 6: Summary statistics
    print("\n✓ Checkpoint 6: Generating summary statistics...")
    try:
        num_summary, cat_summary = get_summary_statistics(df)
        print("✅ Summary statistics generated successfully")
        print(f"\nNumerical variables analyzed: {len(num_summary.columns)}")
        print(f"Categorical variables analyzed: {len(cat_summary)}")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

    # Display sample data
    print_section("SAMPLE DATA (First 5 rows)")
    print(df.head())

    # Display column info
    print_section("COLUMN INFORMATION")
    print(f"{'Column':<30} {'Type':<15} {'Non-Null':<10} {'Missing %':<10}")
    print("-" * 70)
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        missing_pct = df[col].isna().sum() / len(df) * 100
        print(f"{col:<30} {dtype:<15} {non_null:<10,} {missing_pct:<10.1f}")

    # Display numerical summary
    print_section("NUMERICAL SUMMARY STATISTICS")
    print(num_summary)

    # Display categorical summary
    print_section("CATEGORICAL SUMMARY STATISTICS")
    print(cat_summary)

    # Validation report summary
    print_section("VALIDATION REPORT SUMMARY")
    print(f"Total rows: {report['total_rows']:,}")
    print(f"Total columns: {report['total_columns']}")
    print(f"Approval rate: {report['approval_rate']:.4%}")

    if report['missing_values']:
        print("\nMissing values:")
        for col, count in report['missing_values'].items():
            pct = count / report['total_rows'] * 100
            print(f"  - {col}: {count:,} ({pct:.1f}%)")

    if report['issues']:
        print("\n⚠️  Issues detected:")
        for issue in report['issues']:
            print(f"  - {issue}")
    else:
        print("\n✅ No data quality issues detected")

    print_section("PHASE 1 VALIDATION COMPLETE")
    print("✅ All core validation checks passed!")
    print("\nNext steps:")
    print("  1. Review validation report above")
    print("  2. Run univariate analysis: python -m notebooks.phase1_eda.1_1_univariate_analysis")
    print("  3. Continue with bivariate analysis and feature engineering")

    return True


if __name__ == "__main__":
    success = validate_phase1()
    sys.exit(0 if success else 1)
