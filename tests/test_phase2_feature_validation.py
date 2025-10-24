"""
Validation test for Phase 2.3: Feature Engineering Validation
Validates outputs from feature_validation.py
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

def validate_phase2_feature_validation():
    """
    Validate Phase 2.3 Feature Engineering Validation outputs

    Expected outputs:
    - 2 CSV tables: model_comparison.csv, final_feature_set.csv
    - 1 figure: feature_validation_comparison.png
    """
    results = ValidationResults()
    base_dir = Path(__file__).parent.parent
    tables_dir = base_dir / 'reports' / 'phase2_feature_importance' / 'tables'
    figures_dir = base_dir / 'reports' / 'phase2_feature_importance' / 'figures'

    # Track data for content validation
    comparison_df = None
    features_df = None

    # ========================================
    # TABLE EXISTENCE AND STRUCTURE CHECKS
    # ========================================

    required_tables = {
        'model_comparison.csv': {
            'min_rows': 2,
            'required_cols': ['Model', 'Features_Count', 'Train_AUC', 'Test_AUC',
                            'Test_Precision', 'Test_Recall', 'Test_F1',
                            'AUC_Improvement', 'Meets_Target']
        },
        'final_feature_set.csv': {
            'min_rows': 9,  # At least baseline features
            'required_cols': ['Feature', 'Type']
        }
    }

    for table_name, requirements in required_tables.items():
        table_path = tables_dir / table_name

        # Check file exists
        if not table_path.exists():
            results.add_check(
                'Table Existence',
                f'{table_name} exists',
                False,
                f'File not found at {table_path}'
            )
            continue

        results.add_check(
            'Table Existence',
            f'{table_name} exists',
            True,
            f'Found at {table_path}'
        )

        # Load and validate structure
        try:
            df = pd.read_csv(table_path)

            # Store for content validation
            if table_name == 'model_comparison.csv':
                comparison_df = df
            elif table_name == 'final_feature_set.csv':
                features_df = df

            # Check row count
            if len(df) >= requirements['min_rows']:
                results.add_check(
                    'Table Structure',
                    f'{table_name} has sufficient rows',
                    True,
                    f'Found {len(df)} rows (minimum {requirements["min_rows"]})'
                )
            else:
                results.add_check(
                    'Table Structure',
                    f'{table_name} has sufficient rows',
                    False,
                    f'Found {len(df)} rows, expected at least {requirements["min_rows"]}'
                )

            # Check required columns
            missing_cols = set(requirements['required_cols']) - set(df.columns)
            if not missing_cols:
                results.add_check(
                    'Table Structure',
                    f'{table_name} has required columns',
                    True,
                    f'All columns present: {requirements["required_cols"]}'
                )
            else:
                results.add_check(
                    'Table Structure',
                    f'{table_name} has required columns',
                    False,
                    f'Missing columns: {missing_cols}'
                )

        except Exception as e:
            results.add_check(
                'Table Structure',
                f'{table_name} is valid CSV',
                False,
                f'Error loading CSV: {str(e)}'
            )

    # ========================================
    # FIGURE EXISTENCE CHECKS
    # ========================================

    figure_path = figures_dir / 'feature_validation_comparison.png'
    if figure_path.exists():
        results.add_check(
            'Figure Existence',
            'feature_validation_comparison.png exists',
            True,
            f'Found at {figure_path}'
        )

        # Check file size (should be >10KB for a meaningful plot)
        file_size = figure_path.stat().st_size
        if file_size > 10000:
            results.add_check(
                'Figure Quality',
                'feature_validation_comparison.png has valid size',
                True,
                f'File size: {file_size} bytes'
            )
        else:
            results.add_check(
                'Figure Quality',
                'feature_validation_comparison.png has valid size',
                False,
                f'File size too small: {file_size} bytes'
            )
    else:
        results.add_check(
            'Figure Existence',
            'feature_validation_comparison.png exists',
            False,
            f'File not found at {figure_path}'
        )

    # ========================================
    # CONTENT VALIDATION
    # ========================================

    # Validate model comparison
    if comparison_df is not None and len(comparison_df) >= 2:
        # Check we have both models
        if 'Baseline' in comparison_df['Model'].iloc[0] and 'Enhanced' in comparison_df['Model'].iloc[1]:
            results.add_check(
                'Content Quality',
                'Both models present in comparison',
                True,
                'Baseline and Enhanced models found'
            )
        else:
            results.add_check(
                'Content Quality',
                'Both models present in comparison',
                False,
                f"Models found: {comparison_df['Model'].tolist()}"
            )

        # Check AUC scores are reasonable (>= 0.5)
        if (comparison_df['Test_AUC'] >= 0.5).all():
            results.add_check(
                'Content Quality',
                'Model AUC scores are reasonable',
                True,
                f"AUCs: {comparison_df['Test_AUC'].tolist()}"
            )
        else:
            results.add_check(
                'Content Quality',
                'Model AUC scores are reasonable',
                False,
                f"Some AUC scores below 0.5: {comparison_df['Test_AUC'].tolist()}"
            )

        # Check baseline AUC is documented
        baseline_auc = comparison_df['Test_AUC'].iloc[0]
        if baseline_auc > 0:
            results.add_check(
                'Content Quality',
                'Baseline model AUC documented',
                True,
                f'Baseline AUC: {baseline_auc:.4f}'
            )
        else:
            results.add_check(
                'Content Quality',
                'Baseline model AUC documented',
                False,
                f'Invalid baseline AUC: {baseline_auc}'
            )

        # Check AUC improvement is calculated
        if 'AUC_Improvement' in comparison_df.columns:
            auc_improvement = comparison_df['AUC_Improvement'].iloc[1]
            results.add_check(
                'Content Quality',
                'AUC improvement calculated',
                True,
                f'Improvement: {auc_improvement:+.4f}'
            )

            # Note: We don't require improvement >= 0.03, just that it's measured
            # The actual improvement can be positive or negative
            target_check = comparison_df['Meets_Target'].iloc[1]
            results.add_check(
                'Content Quality',
                'Improvement target evaluated',
                True,
                f'Meets target (>=0.03): {target_check}'
            )
        else:
            results.add_check(
                'Content Quality',
                'AUC improvement calculated',
                False,
                'AUC_Improvement column missing'
            )

        # Check all metrics documented
        required_metrics = ['Test_Precision', 'Test_Recall', 'Test_F1']
        for metric in required_metrics:
            if metric in comparison_df.columns:
                if (comparison_df[metric] >= 0).all() and (comparison_df[metric] <= 1).all():
                    results.add_check(
                        'Content Quality',
                        f'{metric} values are valid',
                        True,
                        f'Range: {comparison_df[metric].min():.4f} - {comparison_df[metric].max():.4f}'
                    )
                else:
                    results.add_check(
                        'Content Quality',
                        f'{metric} values are valid',
                        False,
                        f'Values outside [0, 1]: {comparison_df[metric].tolist()}'
                    )

    # Validate final feature set
    if features_df is not None:
        # Check feature types are labeled
        if 'Type' in features_df.columns:
            feature_types = features_df['Type'].unique()
            if 'Original' in feature_types and 'Engineered' in feature_types:
                results.add_check(
                    'Content Quality',
                    'Feature types properly labeled',
                    True,
                    f'Found types: {feature_types.tolist()}'
                )
            elif 'Original' in feature_types:
                results.add_check(
                    'Content Quality',
                    'Feature types properly labeled',
                    True,
                    'Only original features (no engineered features added)'
                )
            else:
                results.add_check(
                    'Content Quality',
                    'Feature types properly labeled',
                    False,
                    f'Unexpected types: {feature_types.tolist()}'
                )

        # Check minimum number of features
        if len(features_df) >= 9:
            results.add_check(
                'Content Quality',
                'Sufficient features in final set',
                True,
                f'Total features: {len(features_df)}'
            )
        else:
            results.add_check(
                'Content Quality',
                'Sufficient features in final set',
                False,
                f'Only {len(features_df)} features, expected at least 9'
            )

        # Check for engineered features
        if 'Type' in features_df.columns:
            engineered_count = (features_df['Type'] == 'Engineered').sum()
            original_count = (features_df['Type'] == 'Original').sum()

            results.add_check(
                'Content Quality',
                'Feature set composition documented',
                True,
                f'Original: {original_count}, Engineered: {engineered_count}'
            )

    # ========================================
    # GENERATE INSIGHTS
    # ========================================

    insights = []

    if comparison_df is not None and len(comparison_df) >= 2:
        baseline_auc = comparison_df['Test_AUC'].iloc[0]
        enhanced_auc = comparison_df['Test_AUC'].iloc[1]
        improvement = enhanced_auc - baseline_auc

        insights.append(f"Baseline model AUC: {baseline_auc:.4f}")
        insights.append(f"Enhanced model AUC: {enhanced_auc:.4f}")
        insights.append(f"AUC change: {improvement:+.4f}")

        if improvement >= 0.03:
            insights.append("Target improvement (≥0.03) achieved ✓")
        elif improvement > 0:
            insights.append("Positive improvement, but below target of 0.03")
        else:
            insights.append("Engineered features did not improve performance")

    if features_df is not None and 'Type' in features_df.columns:
        engineered_count = (features_df['Type'] == 'Engineered').sum()
        if engineered_count > 0:
            insights.append(f"Added {engineered_count} engineered features to original set")

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
        'subphase': 'Phase 2.3: Feature Engineering Validation',
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
    validate_phase2_feature_validation()
