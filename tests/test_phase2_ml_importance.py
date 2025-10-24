"""
Validation test for Phase 2.2: ML Feature Importance
Validates outputs from ml_importance.py
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

def validate_phase2_ml_importance():
    """
    Validate Phase 2.2 ML Feature Importance outputs

    Expected outputs:
    - 5 CSV tables: rf_feature_importance.csv, xgb_feature_importance.csv,
                    lr_feature_importance.csv, consensus_feature_rankings.csv,
                    model_performance.csv
    - 1 figure: ml_feature_importance.png
    """
    results = ValidationResults()
    base_dir = Path(__file__).parent.parent
    tables_dir = base_dir / 'reports' / 'phase2_feature_importance' / 'tables'
    figures_dir = base_dir / 'reports' / 'phase2_feature_importance' / 'figures'

    # Track data for content validation
    rf_df = None
    xgb_df = None
    lr_df = None
    consensus_df = None
    performance_df = None

    # ========================================
    # TABLE EXISTENCE AND STRUCTURE CHECKS
    # ========================================

    required_tables = {
        'rf_feature_importance.csv': {
            'min_rows': 5,
            'required_cols': ['Feature', 'Gini_Importance']
        },
        'xgb_feature_importance.csv': {
            'min_rows': 5,
            'required_cols': ['Feature', 'Gain', 'Cover', 'Weight']
        },
        'lr_feature_importance.csv': {
            'min_rows': 5,
            'required_cols': ['Feature', 'Coefficient', 'Abs_Coefficient', 'Selected']
        },
        'consensus_feature_rankings.csv': {
            'min_rows': 5,
            'required_cols': ['Feature', 'Consensus_Score', 'Rank', 'RF_Score', 'XGB_Score', 'LR_Score']
        },
        'model_performance.csv': {
            'min_rows': 3,
            'required_cols': ['Model', 'Test_AUC', 'Meets_Target']
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
            if table_name == 'rf_feature_importance.csv':
                rf_df = df
            elif table_name == 'xgb_feature_importance.csv':
                xgb_df = df
            elif table_name == 'lr_feature_importance.csv':
                lr_df = df
            elif table_name == 'consensus_feature_rankings.csv':
                consensus_df = df
            elif table_name == 'model_performance.csv':
                performance_df = df

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

    figure_path = figures_dir / 'ml_feature_importance.png'
    if figure_path.exists():
        results.add_check(
            'Figure Existence',
            'ml_feature_importance.png exists',
            True,
            f'Found at {figure_path}'
        )

        # Check file size (should be >10KB for a meaningful plot)
        file_size = figure_path.stat().st_size
        if file_size > 10000:
            results.add_check(
                'Figure Quality',
                'ml_feature_importance.png has valid size',
                True,
                f'File size: {file_size} bytes'
            )
        else:
            results.add_check(
                'Figure Quality',
                'ml_feature_importance.png has valid size',
                False,
                f'File size too small: {file_size} bytes'
            )
    else:
        results.add_check(
            'Figure Existence',
            'ml_feature_importance.png exists',
            False,
            f'File not found at {figure_path}'
        )

    # ========================================
    # CONTENT VALIDATION
    # ========================================

    # Validate Random Forest importance scores
    if rf_df is not None:
        # Check Gini importance scores are valid (0 to 1)
        if (rf_df['Gini_Importance'] >= 0).all() and (rf_df['Gini_Importance'] <= 1).all():
            results.add_check(
                'Content Quality',
                'RF Gini importance scores are valid',
                True,
                'All scores between 0 and 1'
            )
        else:
            results.add_check(
                'Content Quality',
                'RF Gini importance scores are valid',
                False,
                'Some scores outside valid range [0, 1]'
            )

        # Check importances sum to 1 (normalized)
        importance_sum = rf_df['Gini_Importance'].sum()
        if 0.99 <= importance_sum <= 1.01:
            results.add_check(
                'Content Quality',
                'RF Gini importances normalized',
                True,
                f'Sum of importances: {importance_sum:.4f}'
            )
        else:
            results.add_check(
                'Content Quality',
                'RF Gini importances normalized',
                False,
                f'Sum of importances not normalized: {importance_sum:.4f}'
            )

    # Validate XGBoost importance metrics
    if xgb_df is not None:
        # Check all importance metrics are non-negative
        if (xgb_df['Gain'] >= 0).all() and (xgb_df['Cover'] >= 0).all() and (xgb_df['Weight'] >= 0).all():
            results.add_check(
                'Content Quality',
                'XGBoost importance metrics are non-negative',
                True,
                'All Gain, Cover, Weight >= 0'
            )
        else:
            results.add_check(
                'Content Quality',
                'XGBoost importance metrics are non-negative',
                False,
                'Some metrics are negative'
            )

    # Validate Logistic Regression coefficients
    if lr_df is not None:
        # Check selected features exist
        if 'Selected' in lr_df.columns and lr_df['Selected'].sum() > 0:
            results.add_check(
                'Content Quality',
                'LR has selected features',
                True,
                f"{lr_df['Selected'].sum()} features selected by L1 regularization"
            )
        else:
            results.add_check(
                'Content Quality',
                'LR has selected features',
                False,
                'No features selected'
            )

    # Validate Consensus rankings
    if consensus_df is not None:
        # Check consensus scores are valid (0 to 1)
        if (consensus_df['Consensus_Score'] >= 0).all() and (consensus_df['Consensus_Score'] <= 1).all():
            results.add_check(
                'Content Quality',
                'Consensus scores are valid',
                True,
                'All scores between 0 and 1'
            )
        else:
            results.add_check(
                'Content Quality',
                'Consensus scores are valid',
                False,
                'Some scores outside valid range [0, 1]'
            )

        # Check consensus ranks are sequential
        if list(consensus_df['Rank']) == list(range(1, len(consensus_df) + 1)):
            results.add_check(
                'Content Quality',
                'Consensus ranks are sequential',
                True,
                'Ranks properly assigned'
            )
        else:
            results.add_check(
                'Content Quality',
                'Consensus ranks are sequential',
                False,
                'Rank sequence broken'
            )

        # Check top 10 features identified
        top_10 = consensus_df.head(10)
        if len(top_10) == 10:
            results.add_check(
                'Content Quality',
                'Top 10 features identified',
                True,
                f"Top feature: {top_10.iloc[0]['Feature']} (score: {top_10.iloc[0]['Consensus_Score']:.3f})"
            )
        else:
            results.add_check(
                'Content Quality',
                'Top 10 features identified',
                False,
                f'Only {len(top_10)} features in consensus'
            )

    # Validate model performance
    if performance_df is not None:
        # Check all models have AUC scores
        expected_models = ['Random Forest', 'XGBoost', 'Logistic Regression']
        found_models = performance_df['Model'].tolist()

        if set(expected_models) == set(found_models):
            results.add_check(
                'Content Quality',
                'All models evaluated',
                True,
                f'Found all 3 models: {found_models}'
            )
        else:
            results.add_check(
                'Content Quality',
                'All models evaluated',
                False,
                f'Missing models: {set(expected_models) - set(found_models)}'
            )

        # Check AUC scores are reasonable (>= 0.5, ideally >= 0.70)
        if (performance_df['Test_AUC'] >= 0.5).all():
            results.add_check(
                'Content Quality',
                'Model AUC scores are reasonable',
                True,
                f"Min AUC: {performance_df['Test_AUC'].min():.3f}, Max AUC: {performance_df['Test_AUC'].max():.3f}"
            )
        else:
            results.add_check(
                'Content Quality',
                'Model AUC scores are reasonable',
                False,
                f"Some AUC scores below 0.5: {performance_df[performance_df['Test_AUC'] < 0.5]['Model'].tolist()}"
            )

        # Check if models meet target performance (>= 0.70)
        target_met = (performance_df['Test_AUC'] >= 0.70).sum()
        if target_met >= 2:
            results.add_check(
                'Content Quality',
                'Models meet target AUC >= 0.70',
                True,
                f'{target_met}/3 models meet target'
            )
        else:
            results.add_check(
                'Content Quality',
                'Models meet target AUC >= 0.70',
                False,
                f'Only {target_met}/3 models meet target'
            )

    # ========================================
    # GENERATE INSIGHTS
    # ========================================

    insights = []

    if consensus_df is not None and len(consensus_df) >= 10:
        top_feature = consensus_df.iloc[0]['Feature']
        top_score = consensus_df.iloc[0]['Consensus_Score']
        insights.append(f"Top feature by consensus: {top_feature} (score: {top_score:.3f})")

    if performance_df is not None:
        best_model = performance_df.loc[performance_df['Test_AUC'].idxmax()]
        insights.append(f"Best performing model: {best_model['Model']} (AUC: {best_model['Test_AUC']:.3f})")

    if lr_df is not None and 'Selected' in lr_df.columns:
        selected_count = lr_df['Selected'].sum()
        insights.append(f"L1 regularization selected {selected_count}/{len(lr_df)} features")

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
        'subphase': 'Phase 2.2: ML Feature Importance',
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
    validate_phase2_ml_importance()
