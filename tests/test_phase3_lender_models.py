"""
Validation test for Phase 3.2: Lender-Specific Predictive Models
Validates outputs from lender_models.py
"""

import os
import json
import pickle
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

def validate_phase3_lender_models():
    """
    Validate Phase 3.2 Lender-Specific Predictive Models outputs

    Expected outputs:
    - 3 trained model files (.pkl)
    - 3 metadata JSON files
    - 3 feature importance CSVs
    - 1 performance summary table
    - 1 comparison visualization
    """
    results = ValidationResults()
    base_dir = Path(__file__).parent.parent
    models_dir = base_dir / 'models' / 'phase3_lender_models'
    tables_dir = base_dir / 'reports' / 'phase3_lender_analysis' / 'tables'
    figures_dir = base_dir / 'reports' / 'phase3_lender_analysis' / 'figures'

    lenders = ['a', 'b', 'c']  # lowercase for filenames
    lender_names = ['A', 'B', 'C']

    # Track data for validation
    model_metadata = {}
    feature_importance_dfs = {}
    performance_df = None

    # ========================================
    # MODEL FILE CHECKS
    # ========================================

    for lender, lender_name in zip(lenders, lender_names):
        model_path = models_dir / f'lender_{lender}_model.pkl'

        # Check model file exists
        if model_path.exists():
            results.add_check(
                'Model Files',
                f'lender_{lender}_model.pkl exists',
                True,
                f'Found model for Lender {lender_name}'
            )

            # Try to load model
            try:
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)

                # Check model structure
                if 'model' in model_data and 'feature_names' in model_data:
                    results.add_check(
                        'Model Structure',
                        f'Lender {lender_name} model has required components',
                        True,
                        'Model and feature names present'
                    )

                    # Check metrics
                    if 'metrics' in model_data:
                        test_auc = model_data['metrics'].get('test_auc', 0)
                        if test_auc >= 0.70:
                            results.add_check(
                                'Model Performance',
                                f'Lender {lender_name} meets AUC target',
                                True,
                                f'Test AUC: {test_auc:.4f}'
                            )
                        else:
                            results.add_check(
                                'Model Performance',
                                f'Lender {lender_name} meets AUC target',
                                False,
                                f'Test AUC below 0.70: {test_auc:.4f}'
                            )
                else:
                    results.add_check(
                        'Model Structure',
                        f'Lender {lender_name} model has required components',
                        False,
                        'Missing model or feature_names'
                    )
            except Exception as e:
                results.add_check(
                    'Model Loading',
                    f'Lender {lender_name} model loads successfully',
                    False,
                    f'Error: {str(e)}'
                )
        else:
            results.add_check(
                'Model Files',
                f'lender_{lender}_model.pkl exists',
                False,
                f'Model file not found for Lender {lender_name}'
            )

        # Check metadata JSON
        metadata_path = models_dir / f'lender_{lender}_metadata.json'
        if metadata_path.exists():
            results.add_check(
                'Metadata Files',
                f'lender_{lender}_metadata.json exists',
                True,
                f'Found metadata for Lender {lender_name}'
            )

            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    model_metadata[lender_name] = metadata

                # Check required fields
                required_fields = ['lender', 'best_params', 'test_auc', 'cv_score']
                if all(field in metadata for field in required_fields):
                    results.add_check(
                        'Metadata Content',
                        f'Lender {lender_name} metadata has required fields',
                        True,
                        'All required fields present'
                    )

                    # Check hyperparameters documented
                    if metadata['best_params']:
                        results.add_check(
                            'Hyperparameters',
                            f'Lender {lender_name} hyperparameters documented',
                            True,
                            f"Params: {list(metadata['best_params'].keys())}"
                        )
                else:
                    results.add_check(
                        'Metadata Content',
                        f'Lender {lender_name} metadata has required fields',
                        False,
                        f'Missing fields: {set(required_fields) - set(metadata.keys())}'
                    )
            except Exception as e:
                results.add_check(
                    'Metadata Loading',
                    f'Lender {lender_name} metadata loads successfully',
                    False,
                    f'Error: {str(e)}'
                )
        else:
            results.add_check(
                'Metadata Files',
                f'lender_{lender}_metadata.json exists',
                False,
                f'Metadata file not found for Lender {lender_name}'
            )

    # ========================================
    # TABLE CHECKS
    # ========================================

    # Feature importance tables
    for lender_name in lender_names:
        fi_path = tables_dir / f'lender_{lender_name}_feature_importance.csv'

        if fi_path.exists():
            results.add_check(
                'Table Existence',
                f'lender_{lender_name}_feature_importance.csv exists',
                True,
                f'Found feature importance for Lender {lender_name}'
            )

            try:
                fi_df = pd.read_csv(fi_path)
                feature_importance_dfs[lender_name] = fi_df

                # Check structure
                if 'Feature' in fi_df.columns and 'Importance' in fi_df.columns:
                    results.add_check(
                        'Table Structure',
                        f'Lender {lender_name} feature importance has required columns',
                        True,
                        'Feature and Importance columns present'
                    )

                    # Check has features
                    if len(fi_df) >= 5:
                        results.add_check(
                            'Content Quality',
                            f'Lender {lender_name} has sufficient features',
                            True,
                            f'{len(fi_df)} features documented'
                        )
            except Exception as e:
                results.add_check(
                    'Table Loading',
                    f'Lender {lender_name} feature importance loads successfully',
                    False,
                    f'Error: {str(e)}'
                )
        else:
            results.add_check(
                'Table Existence',
                f'lender_{lender_name}_feature_importance.csv exists',
                False,
                f'Feature importance table not found for Lender {lender_name}'
            )

    # Performance summary table
    perf_path = tables_dir / 'lender_models_performance.csv'
    if perf_path.exists():
        results.add_check(
            'Table Existence',
            'lender_models_performance.csv exists',
            True,
            'Found performance summary'
        )

        try:
            performance_df = pd.read_csv(perf_path)

            # Check structure
            required_cols = ['Lender', 'Test_AUC', 'CV_AUC', 'Meets_Target_070']
            if all(col in performance_df.columns for col in required_cols):
                results.add_check(
                    'Table Structure',
                    'Performance summary has required columns',
                    True,
                    'All required columns present'
                )

                # Check all lenders present
                if set(performance_df['Lender']) == set(lender_names):
                    results.add_check(
                        'Content Quality',
                        'All lenders in performance summary',
                        True,
                        'A, B, C all present'
                    )

                    # Check all meet target
                    if performance_df['Meets_Target_070'].all():
                        results.add_check(
                            'Model Performance',
                            'All models meet AUC target',
                            True,
                            'All models have AUC >= 0.70'
                        )
                    else:
                        failing = performance_df[~performance_df['Meets_Target_070']]['Lender'].tolist()
                        results.add_check(
                            'Model Performance',
                            'All models meet AUC target',
                            False,
                            f'Models not meeting target: {failing}'
                        )
        except Exception as e:
            results.add_check(
                'Table Loading',
                'Performance summary loads successfully',
                False,
                f'Error: {str(e)}'
            )
    else:
        results.add_check(
            'Table Existence',
            'lender_models_performance.csv exists',
            False,
            'Performance summary not found'
        )

    # ========================================
    # FIGURE CHECKS
    # ========================================

    comparison_path = figures_dir / 'lender_models_comparison.png'
    if comparison_path.exists():
        results.add_check(
            'Figure Existence',
            'lender_models_comparison.png exists',
            True,
            'Found comparison visualization'
        )

        file_size = comparison_path.stat().st_size
        if file_size > 10000:
            results.add_check(
                'Figure Quality',
                'lender_models_comparison.png has valid size',
                True,
                f'File size: {file_size} bytes'
            )
    else:
        results.add_check(
            'Figure Existence',
            'lender_models_comparison.png exists',
            False,
            'Comparison visualization not found'
        )

    # ========================================
    # CONTENT VALIDATION
    # ========================================

    # Check feature importance differs between lenders
    if len(feature_importance_dfs) == 3:
        top_features = {}
        for lender, fi_df in feature_importance_dfs.items():
            top_features[lender] = fi_df.head(3)['Feature'].tolist()

        # Check if top features differ
        all_same = (top_features['A'] == top_features['B'] == top_features['C'])
        if not all_same:
            results.add_check(
                'Content Quality',
                'Feature importance differs between lenders',
                True,
                'Top features vary across lenders'
            )
        else:
            results.add_check(
                'Content Quality',
                'Feature importance differs between lenders',
                False,
                'All lenders have identical top features'
            )

    # Check model performance consistency with metadata
    if performance_df is not None and model_metadata:
        for lender_name in lender_names:
            if lender_name in model_metadata:
                meta_auc = model_metadata[lender_name].get('test_auc')
                perf_row = performance_df[performance_df['Lender'] == lender_name]

                if not perf_row.empty:
                    perf_auc = perf_row['Test_AUC'].values[0]
                    if abs(meta_auc - perf_auc) < 0.001:  # Allow small floating point diff
                        results.add_check(
                            'Data Consistency',
                            f'Lender {lender_name} AUC matches across files',
                            True,
                            f'Consistent: {meta_auc:.4f}'
                        )

    # ========================================
    # GENERATE INSIGHTS
    # ========================================

    insights = []

    if performance_df is not None:
        for _, row in performance_df.iterrows():
            insights.append(f"Lender {row['Lender']}: Test AUC = {row['Test_AUC']:.4f}")

        avg_auc = performance_df['Test_AUC'].mean()
        insights.append(f"Average AUC across lenders: {avg_auc:.4f}")

        meets_target_count = performance_df['Meets_Target_070'].sum()
        insights.append(f"{meets_target_count}/3 models meet AUC >= 0.70 target")

    if feature_importance_dfs:
        for lender, fi_df in feature_importance_dfs.items():
            top_feature = fi_df.iloc[0]['Feature']
            insights.append(f"Lender {lender} top feature: {top_feature}")

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
        'subphase': 'Phase 3.2: Lender-Specific Predictive Models',
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
    validate_phase3_lender_models()
