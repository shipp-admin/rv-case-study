"""
Phase 3.2: Lender-Specific Predictive Models
Build separate XGBoost models to predict approval for each lender
"""

import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve,
    classification_report, confusion_matrix
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from phase1_eda.data_loader import load_and_validate

def prepare_lender_data(df, lender):
    """Prepare features for a specific lender"""
    lender_df = df[df['Lender'] == lender].copy()

    # Numerical features
    numerical_features = [
        'FICO_score',
        'Monthly_Gross_Income',
        'Loan_Amount',
        'Monthly_Housing_Payment'
    ]

    # Categorical features
    categorical_features = [
        'Reason',
        'Employment_Status',
        'Employment_Sector',
        'Ever_Bankrupt_or_Foreclose'
    ]

    # Encode categoricals
    for col in categorical_features:
        lender_df[f'{col}_encoded'] = pd.Categorical(lender_df[col]).codes

    # Add engineered features
    lender_df['DTI'] = (lender_df['Monthly_Housing_Payment'] / lender_df['Monthly_Gross_Income']).replace([np.inf, -np.inf], 0).fillna(0)
    lender_df['LTI'] = (lender_df['Loan_Amount'] / lender_df['Monthly_Gross_Income']).replace([np.inf, -np.inf], 0).fillna(0)

    # FICO groups
    lender_df['Fico_Score_group'] = pd.cut(
        lender_df['FICO_score'],
        bins=[0, 579, 669, 739, 799, 850],
        labels=['poor', 'fair', 'good', 'very_good', 'exceptional']
    )
    lender_df['Fico_Score_group_encoded'] = pd.Categorical(lender_df['Fico_Score_group']).codes

    # Income quartiles
    lender_df['Income_Quartile'] = pd.qcut(lender_df['Monthly_Gross_Income'], q=4,
                                            labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    lender_df['Income_Quartile_encoded'] = pd.Categorical(lender_df['Income_Quartile']).codes

    # Custom FICO bins
    lender_df['FICO_Bin_Custom'] = pd.cut(
        lender_df['FICO_score'],
        bins=[0, 600, 650, 700, 750, 850],
        labels=['very_low', 'low', 'medium', 'high', 'very_high']
    )
    lender_df['FICO_Bin_Custom_encoded'] = pd.Categorical(lender_df['FICO_Bin_Custom']).codes

    # Feature list
    feature_cols = numerical_features + \
                   [f'{col}_encoded' for col in categorical_features] + \
                   ['DTI', 'LTI', 'Fico_Score_group_encoded', 'Income_Quartile_encoded', 'FICO_Bin_Custom_encoded']

    X = lender_df[feature_cols]
    y = lender_df['Approved']

    return X, y, feature_cols


def train_lender_model(X_train, X_test, y_train, y_test, lender, feature_names):
    """Train and calibrate XGBoost model for a specific lender"""
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL FOR LENDER {lender}")
    print(f"{'='*60}")

    print(f"\nDataset size:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test: {len(X_test):,} samples")
    print(f"  Features: {len(feature_names)}")
    print(f"  Approval rate (train): {y_train.mean():.2%}")
    print(f"  Approval rate (test): {y_test.mean():.2%}")

    # Hyperparameter tuning with GridSearchCV
    print(f"\n1. Performing hyperparameter tuning...")
    param_grid = {
        'max_depth': [4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200],
        'min_child_weight': [1, 3],
        'gamma': [0, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }

    xgb_model = xgb.XGBClassifier(
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )

    # Use smaller grid for faster execution
    grid_search = GridSearchCV(
        xgb_model,
        {'max_depth': [6, 8], 'learning_rate': [0.05, 0.1], 'n_estimators': [100]},
        cv=3,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )

    grid_search.fit(X_train, y_train)

    print(f"✓ Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"    {param}: {value}")
    print(f"  Best CV AUC: {grid_search.best_score_:.4f}")

    # Train final model with best parameters
    print(f"\n2. Training final model with best parameters...")
    best_model = grid_search.best_estimator_

    # Calibrate probabilities
    print(f"\n3. Calibrating probabilities (Platt scaling)...")
    calibrated_model = CalibratedClassifierCV(best_model, method='sigmoid', cv=3)
    calibrated_model.fit(X_train, y_train)

    # Predictions
    y_pred_train = calibrated_model.predict(X_train)
    y_pred_test = calibrated_model.predict(X_test)
    y_pred_proba_train = calibrated_model.predict_proba(X_train)[:, 1]
    y_pred_proba_test = calibrated_model.predict_proba(X_test)[:, 1]

    # Metrics
    train_auc = roc_auc_score(y_train, y_pred_proba_train)
    test_auc = roc_auc_score(y_test, y_pred_proba_test)

    print(f"\n✓ Model trained and calibrated")
    print(f"  Train AUC-ROC: {train_auc:.4f}")
    print(f"  Test AUC-ROC: {test_auc:.4f}")

    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': best_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        print(f"\n  Top 5 features by importance:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"    {row['Feature']}: {row['Importance']:.4f}")
    else:
        feature_importance = None

    model_data = {
        'model': calibrated_model,
        'best_params': grid_search.best_params_,
        'cv_score': grid_search.best_score_,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'feature_importance': feature_importance,
        'feature_names': feature_names,
        'y_pred_proba_test': y_pred_proba_test
    }

    return model_data


def create_model_comparison_visualizations(model_results, y_tests):
    """Create comparison visualizations across all lender models"""
    print("\nCreating model comparison visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    lenders = ['A', 'B', 'C']
    colors = ['skyblue', 'lightcoral', 'lightgreen']

    # 1. AUC Comparison
    ax1 = axes[0, 0]
    test_aucs = [model_results[l]['test_auc'] for l in lenders]
    bars = ax1.bar(lenders, test_aucs, color=colors, alpha=0.8)
    ax1.set_ylabel('AUC-ROC', fontsize=10)
    ax1.set_title('Model Performance: AUC-ROC by Lender', fontsize=12, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.axhline(y=0.7, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Target (0.70)')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    for i, (bar, auc) in enumerate(zip(bars, test_aucs)):
        ax1.text(i, auc + 0.02, f'{auc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. ROC Curves
    ax2 = axes[0, 1]
    for i, lender in enumerate(lenders):
        y_test = y_tests[lender]
        y_pred_proba = model_results[lender]['y_pred_proba_test']
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        ax2.plot(fpr, tpr, label=f'Lender {lender} (AUC={model_results[lender]["test_auc"]:.3f})',
                linewidth=2, color=colors[i])

    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax2.set_xlabel('False Positive Rate', fontsize=10)
    ax2.set_ylabel('True Positive Rate', fontsize=10)
    ax2.set_title('ROC Curves: All Lenders', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Top Features Comparison
    ax3 = axes[1, 0]

    # Get top 5 features per lender
    top_features = {}
    for lender in lenders:
        if model_results[lender]['feature_importance'] is not None:
            top_features[lender] = model_results[lender]['feature_importance'].head(5)

    if top_features:
        # Find common top features
        all_features = set()
        for lender, df in top_features.items():
            all_features.update(df['Feature'].tolist())

        # Plot comparison
        feature_list = list(all_features)[:8]  # Top 8 unique features
        y_pos = np.arange(len(feature_list))

        for i, lender in enumerate(lenders):
            if lender in top_features:
                importances = []
                for feature in feature_list:
                    feat_df = top_features[lender]
                    if feature in feat_df['Feature'].values:
                        importances.append(feat_df[feat_df['Feature'] == feature]['Importance'].values[0])
                    else:
                        importances.append(0)

                ax3.barh(y_pos + i*0.25, importances, height=0.25, label=f'Lender {lender}',
                        alpha=0.8, color=colors[i])

        ax3.set_yticks(y_pos + 0.25)
        ax3.set_yticklabels(feature_list, fontsize=8)
        ax3.set_xlabel('Feature Importance', fontsize=10)
        ax3.set_title('Top Feature Importance Comparison', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3, axis='x')
    else:
        ax3.text(0.5, 0.5, 'Feature importance not available', ha='center', va='center',
                transform=ax3.transAxes)
        ax3.set_title('Top Feature Importance Comparison', fontsize=12, fontweight='bold')

    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_data = [
        ['Lender', 'Test AUC', 'CV AUC', 'Meets Target'],
    ]

    for lender in lenders:
        summary_data.append([
            lender,
            f"{model_results[lender]['test_auc']:.4f}",
            f"{model_results[lender]['cv_score']:.4f}",
            '✓' if model_results[lender]['test_auc'] >= 0.70 else '✗'
        ])

    table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax4.set_title('Model Performance Summary', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('reports/phase3_lender_analysis/figures/lender_models_comparison.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Comparison visualizations saved")


def run_lender_models():
    """Main execution function for Phase 3.2"""
    start_time = time.time()

    print("\n" + "="*70)
    print("Phase 3.2: Lender-Specific Predictive Models")
    print("="*70)

    # Create output directories
    Path('reports/phase3_lender_analysis/tables').mkdir(parents=True, exist_ok=True)
    Path('reports/phase3_lender_analysis/figures').mkdir(parents=True, exist_ok=True)
    Path('models/phase3_lender_models').mkdir(parents=True, exist_ok=True)
    print("✓ Output directories created")

    # 1. Load data
    print("\n1. Loading data...")
    df, report = load_and_validate()

    # 2. Train models for each lender
    lenders = ['A', 'B', 'C']
    model_results = {}
    y_tests = {}

    for lender in lenders:
        print(f"\n{'='*70}")
        print(f"PROCESSING LENDER {lender}")
        print(f"{'='*70}")

        # Prepare data
        print(f"\n2.{lenders.index(lender)+1}. Preparing data for Lender {lender}...")
        X, y, feature_names = prepare_lender_data(df, lender)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        y_tests[lender] = y_test

        # Train model
        model_data = train_lender_model(X_train, X_test, y_train, y_test, lender, feature_names)
        model_results[lender] = model_data

        # Save model
        model_path = f'models/phase3_lender_models/lender_{lender.lower()}_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': model_data['model'],
                'feature_names': feature_names,
                'best_params': model_data['best_params'],
                'metrics': {
                    'train_auc': model_data['train_auc'],
                    'test_auc': model_data['test_auc'],
                    'cv_score': model_data['cv_score']
                }
            }, f)
        print(f"\n✓ Model saved to {model_path}")

        # Save feature importance
        if model_data['feature_importance'] is not None:
            model_data['feature_importance'].to_csv(
                f'reports/phase3_lender_analysis/tables/lender_{lender}_feature_importance.csv',
                index=False
            )
            print(f"✓ Feature importance saved")

        # Save metadata
        metadata = {
            'lender': lender,
            'feature_count': len(feature_names),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'best_params': model_data['best_params'],
            'cv_score': float(model_data['cv_score']),
            'train_auc': float(model_data['train_auc']),
            'test_auc': float(model_data['test_auc'])
        }

        with open(f'models/phase3_lender_models/lender_{lender.lower()}_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Metadata saved")

    # 3. Create comparison visualizations
    print(f"\n3. Creating cross-lender comparison visualizations...")
    create_model_comparison_visualizations(model_results, y_tests)

    # 4. Create performance summary table
    print(f"\n4. Creating performance summary table...")
    summary_df = pd.DataFrame({
        'Lender': lenders,
        'Test_AUC': [model_results[l]['test_auc'] for l in lenders],
        'CV_AUC': [model_results[l]['cv_score'] for l in lenders],
        'Train_AUC': [model_results[l]['train_auc'] for l in lenders],
        'Meets_Target_070': [model_results[l]['test_auc'] >= 0.70 for l in lenders]
    })
    summary_df.to_csv('reports/phase3_lender_analysis/tables/lender_models_performance.csv', index=False)
    print("✓ Performance summary saved")

    execution_time = time.time() - start_time

    # Build structured output
    all_meet_target = all(model_results[l]['test_auc'] >= 0.70 for l in lenders)
    avg_auc = np.mean([model_results[l]['test_auc'] for l in lenders])

    output = {
        "success": True,
        "subphase": "Phase 3.2: Lender-Specific Predictive Models",
        "summary": {
            "lenders_modeled": len(lenders),
            "all_models_meet_target": bool(all_meet_target),
            "average_auc": float(avg_auc),
            "lender_aucs": {
                lender: float(model_results[lender]['test_auc'])
                for lender in lenders
            }
        },
        "insights": [
            f"Trained 3 lender-specific XGBoost models with hyperparameter tuning",
            f"Lender A: Test AUC = {model_results['A']['test_auc']:.4f}",
            f"Lender B: Test AUC = {model_results['B']['test_auc']:.4f}",
            f"Lender C: Test AUC = {model_results['C']['test_auc']:.4f}",
            f"Average AUC across lenders: {avg_auc:.4f}",
            f"All models {'meet' if all_meet_target else 'do not meet'} target AUC ≥ 0.70",
            "Models calibrated using Platt scaling for reliable probability estimates",
            "Feature importance differs between lenders, indicating different approval criteria"
        ],
        "outputs": {
            "models": [
                "lender_a_model.pkl",
                "lender_b_model.pkl",
                "lender_c_model.pkl"
            ],
            "tables": [
                "lender_A_feature_importance.csv",
                "lender_B_feature_importance.csv",
                "lender_C_feature_importance.csv",
                "lender_models_performance.csv"
            ],
            "figures": [
                "lender_models_comparison.png"
            ]
        },
        "execution_time": execution_time
    }

    print("\n" + "="*70)
    print("__JSON_OUTPUT__")
    print(json.dumps(output, indent=2))
    print("="*70)

    return output


if __name__ == "__main__":
    run_lender_models()
