"""
Phase 2.3: Feature Engineering Validation
Compare baseline model (original features) vs enhanced model (with engineered features)
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from phase1_eda.data_loader import load_and_validate

def prepare_baseline_features(df):
    """Prepare baseline features (original features only)"""
    print("\n" + "="*60)
    print("BASELINE FEATURE PREPARATION")
    print("="*60)

    # Original numerical features
    numerical_features = [
        'FICO_score',
        'Monthly_Gross_Income',
        'Loan_Amount',
        'Monthly_Housing_Payment'
    ]

    # Original categorical features
    categorical_features = [
        'Reason',
        'Employment_Status',
        'Employment_Sector',
        'Lender',
        'Ever_Bankrupt_or_Foreclose'
    ]

    # Encode categorical variables
    df_encoded = df.copy()
    encoders = {}

    for col in categorical_features:
        df_encoded[f'{col}_encoded'] = pd.Categorical(df_encoded[col]).codes
        encoders[col] = dict(enumerate(df_encoded[col].astype('category').cat.categories))

    # Select feature columns
    feature_cols = numerical_features + [f'{col}_encoded' for col in categorical_features]

    print(f"\n✓ Baseline features prepared")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Numerical: {len(numerical_features)}")
    print(f"  Categorical (encoded): {len(categorical_features)}")

    return df_encoded, feature_cols, numerical_features, categorical_features, encoders


def prepare_engineered_features(df):
    """Prepare features with engineered additions"""
    print("\n" + "="*60)
    print("ENGINEERED FEATURE PREPARATION")
    print("="*60)

    df_eng = df.copy()

    # Original numerical features
    numerical_features = [
        'FICO_score',
        'Monthly_Gross_Income',
        'Loan_Amount',
        'Monthly_Housing_Payment'
    ]

    # Original categorical features
    categorical_features = [
        'Reason',
        'Employment_Status',
        'Employment_Sector',
        'Lender',
        'Ever_Bankrupt_or_Foreclose'
    ]

    # Encode categorical
    encoders = {}
    for col in categorical_features:
        df_eng[f'{col}_encoded'] = pd.Categorical(df_eng[col]).codes
        encoders[col] = dict(enumerate(df_eng[col].astype('category').cat.categories))

    # Add engineered features
    engineered_features = []

    # DTI ratio
    df_eng['DTI'] = (df_eng['Monthly_Housing_Payment'] / df_eng['Monthly_Gross_Income']).replace([np.inf, -np.inf], 0).fillna(0)
    engineered_features.append('DTI')

    # LTI ratio
    df_eng['LTI'] = (df_eng['Loan_Amount'] / df_eng['Monthly_Gross_Income']).replace([np.inf, -np.inf], 0).fillna(0)
    engineered_features.append('LTI')

    # FICO groups
    df_eng['Fico_Score_group'] = pd.cut(
        df_eng['FICO_score'],
        bins=[0, 579, 669, 739, 799, 850],
        labels=['poor', 'fair', 'good', 'very_good', 'exceptional']
    )
    df_eng['Fico_Score_group_encoded'] = pd.Categorical(df_eng['Fico_Score_group']).codes
    engineered_features.append('Fico_Score_group_encoded')

    # Income quartiles
    df_eng['Income_Quartile'] = pd.qcut(df_eng['Monthly_Gross_Income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
    df_eng['Income_Quartile_encoded'] = pd.Categorical(df_eng['Income_Quartile']).codes
    engineered_features.append('Income_Quartile_encoded')

    # Custom FICO bins
    df_eng['FICO_Bin_Custom'] = pd.cut(
        df_eng['FICO_score'],
        bins=[0, 600, 650, 700, 750, 850],
        labels=['very_low', 'low', 'medium', 'high', 'very_high']
    )
    df_eng['FICO_Bin_Custom_encoded'] = pd.Categorical(df_eng['FICO_Bin_Custom']).codes
    engineered_features.append('FICO_Bin_Custom_encoded')

    # Combined feature set
    baseline_cols = numerical_features + [f'{col}_encoded' for col in categorical_features]
    all_feature_cols = baseline_cols + engineered_features

    print(f"\n✓ Engineered features prepared")
    print(f"  Baseline features: {len(baseline_cols)}")
    print(f"  Engineered features: {len(engineered_features)}")
    print(f"  Total features: {len(all_feature_cols)}")
    print(f"\n  Engineered features added: {', '.join(engineered_features)}")

    return df_eng, all_feature_cols, baseline_cols, engineered_features, encoders


def train_model(X_train, X_test, y_train, y_test, model_name):
    """Train Random Forest classifier and return metrics"""
    print(f"\nTraining {model_name}...")

    # Train model
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )

    rf.fit(X_train, y_train)

    # Predictions
    y_pred_train = rf.predict(X_train)
    y_pred_test = rf.predict(X_test)
    y_pred_proba_train = rf.predict_proba(X_train)[:, 1]
    y_pred_proba_test = rf.predict_proba(X_test)[:, 1]

    # Metrics
    metrics = {
        'train_auc': roc_auc_score(y_train, y_pred_proba_train),
        'test_auc': roc_auc_score(y_test, y_pred_proba_test),
        'train_precision': precision_score(y_train, y_pred_train, zero_division=0),
        'test_precision': precision_score(y_test, y_pred_test, zero_division=0),
        'train_recall': recall_score(y_train, y_pred_train, zero_division=0),
        'test_recall': recall_score(y_test, y_pred_test, zero_division=0),
        'train_f1': f1_score(y_train, y_pred_train, zero_division=0),
        'test_f1': f1_score(y_test, y_pred_test, zero_division=0)
    }

    print(f"✓ {model_name} trained successfully")
    print(f"  Test AUC-ROC: {metrics['test_auc']:.4f}")
    print(f"  Test Precision: {metrics['test_precision']:.4f}")
    print(f"  Test Recall: {metrics['test_recall']:.4f}")
    print(f"  Test F1-Score: {metrics['test_f1']:.4f}")

    return rf, metrics, y_pred_proba_test


def create_comparison_visualization(baseline_metrics, engineered_metrics, baseline_proba, engineered_proba, y_test):
    """Create comparison visualizations"""
    print("\nCreating comparison visualizations...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Metrics comparison
    ax1 = axes[0, 0]
    metrics_names = ['AUC-ROC', 'Precision', 'Recall', 'F1-Score']
    baseline_vals = [
        baseline_metrics['test_auc'],
        baseline_metrics['test_precision'],
        baseline_metrics['test_recall'],
        baseline_metrics['test_f1']
    ]
    engineered_vals = [
        engineered_metrics['test_auc'],
        engineered_metrics['test_precision'],
        engineered_metrics['test_recall'],
        engineered_metrics['test_f1']
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    ax1.bar(x - width/2, baseline_vals, width, label='Baseline', alpha=0.8, color='skyblue')
    ax1.bar(x + width/2, engineered_vals, width, label='With Engineered Features', alpha=0.8, color='lightcoral')
    ax1.set_xlabel('Metric', fontsize=10)
    ax1.set_ylabel('Score', fontsize=10)
    ax1.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names, fontsize=9)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])

    # Add value labels on bars
    for i, (b, e) in enumerate(zip(baseline_vals, engineered_vals)):
        ax1.text(i - width/2, b + 0.02, f'{b:.3f}', ha='center', va='bottom', fontsize=8)
        ax1.text(i + width/2, e + 0.02, f'{e:.3f}', ha='center', va='bottom', fontsize=8)

    # 2. ROC Curves
    ax2 = axes[0, 1]

    # Baseline ROC
    fpr_base, tpr_base, _ = roc_curve(y_test, baseline_proba)
    ax2.plot(fpr_base, tpr_base, label=f'Baseline (AUC={baseline_metrics["test_auc"]:.3f})', linewidth=2, color='skyblue')

    # Engineered ROC
    fpr_eng, tpr_eng, _ = roc_curve(y_test, engineered_proba)
    ax2.plot(fpr_eng, tpr_eng, label=f'Engineered (AUC={engineered_metrics["test_auc"]:.3f})', linewidth=2, color='lightcoral')

    # Diagonal
    ax2.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')

    ax2.set_xlabel('False Positive Rate', fontsize=10)
    ax2.set_ylabel('True Positive Rate', fontsize=10)
    ax2.set_title('ROC Curves Comparison', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # 3. Improvement metrics
    ax3 = axes[1, 0]
    improvements = {
        'AUC-ROC': ((engineered_metrics['test_auc'] - baseline_metrics['test_auc']) / baseline_metrics['test_auc']) * 100,
        'Precision': ((engineered_metrics['test_precision'] - baseline_metrics['test_precision']) / baseline_metrics['test_precision']) * 100 if baseline_metrics['test_precision'] > 0 else 0,
        'Recall': ((engineered_metrics['test_recall'] - baseline_metrics['test_recall']) / baseline_metrics['test_recall']) * 100 if baseline_metrics['test_recall'] > 0 else 0,
        'F1-Score': ((engineered_metrics['test_f1'] - baseline_metrics['test_f1']) / baseline_metrics['test_f1']) * 100 if baseline_metrics['test_f1'] > 0 else 0
    }

    colors = ['green' if v > 0 else 'red' for v in improvements.values()]
    bars = ax3.barh(list(improvements.keys()), list(improvements.values()), color=colors, alpha=0.7)
    ax3.set_xlabel('Improvement (%)', fontsize=10)
    ax3.set_title('Relative Improvement with Engineered Features', fontsize=12, fontweight='bold')
    ax3.axvline(0, color='black', linewidth=0.8)
    ax3.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for i, (k, v) in enumerate(improvements.items()):
        ax3.text(v + 0.2 if v > 0 else v - 0.2, i, f'{v:+.2f}%', va='center', fontsize=9)

    # 4. Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_data = [
        ['Metric', 'Baseline', 'Engineered', 'Δ Absolute', 'Δ %'],
        ['AUC-ROC', f"{baseline_metrics['test_auc']:.4f}", f"{engineered_metrics['test_auc']:.4f}",
         f"{engineered_metrics['test_auc'] - baseline_metrics['test_auc']:+.4f}",
         f"{improvements['AUC-ROC']:+.2f}%"],
        ['Precision', f"{baseline_metrics['test_precision']:.4f}", f"{engineered_metrics['test_precision']:.4f}",
         f"{engineered_metrics['test_precision'] - baseline_metrics['test_precision']:+.4f}",
         f"{improvements['Precision']:+.2f}%"],
        ['Recall', f"{baseline_metrics['test_recall']:.4f}", f"{engineered_metrics['test_recall']:.4f}",
         f"{engineered_metrics['test_recall'] - baseline_metrics['test_recall']:+.4f}",
         f"{improvements['Recall']:+.2f}%"],
        ['F1-Score', f"{baseline_metrics['test_f1']:.4f}", f"{engineered_metrics['test_f1']:.4f}",
         f"{engineered_metrics['test_f1'] - baseline_metrics['test_f1']:+.4f}",
         f"{improvements['F1-Score']:+.2f}%"]
    ]

    table = ax4.table(cellText=summary_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    ax4.set_title('Performance Summary', fontsize=12, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig('reports/phase2_feature_importance/figures/feature_validation_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Visualizations saved")


def run_feature_validation():
    """Main execution function for Phase 2.3"""
    start_time = time.time()

    print("\n" + "="*70)
    print("Phase 2.3: Feature Engineering Validation")
    print("="*70)

    # Create output directories
    Path('reports/phase2_feature_importance/tables').mkdir(parents=True, exist_ok=True)
    Path('reports/phase2_feature_importance/figures').mkdir(parents=True, exist_ok=True)
    print("✓ Output directories created")

    # 1. Load data
    print("\n1. Loading data...")
    df, report = load_and_validate()

    # 2. Prepare baseline features
    print("\n2. Preparing baseline features...")
    df_baseline, baseline_cols, num_baseline, cat_baseline, encoders_baseline = prepare_baseline_features(df)

    # 3. Prepare engineered features
    print("\n3. Preparing engineered features...")
    df_engineered, engineered_cols, baseline_subset, engineered_added, encoders_eng = prepare_engineered_features(df)

    # 4. Split data (same split for both)
    print("\n4. Splitting data...")
    X_baseline = df_baseline[baseline_cols]
    X_engineered = df_engineered[engineered_cols]
    y = df['Approved']

    # Use same random state for fair comparison
    X_base_train, X_base_test, y_train, y_test = train_test_split(
        X_baseline, y, test_size=0.3, random_state=42, stratify=y
    )
    X_eng_train, X_eng_test, _, _ = train_test_split(
        X_engineered, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"✓ Data split complete")
    print(f"  Train set: {len(X_base_train):,} samples")
    print(f"  Test set: {len(X_base_test):,} samples")

    # 5. Train baseline model
    print("\n" + "="*60)
    print("BASELINE MODEL TRAINING")
    print("="*60)
    baseline_model, baseline_metrics, baseline_proba = train_model(
        X_base_train, X_base_test, y_train, y_test, "Baseline Model"
    )

    # 6. Train engineered model
    print("\n" + "="*60)
    print("ENGINEERED FEATURES MODEL TRAINING")
    print("="*60)
    engineered_model, engineered_metrics, engineered_proba = train_model(
        X_eng_train, X_eng_test, y_train, y_test, "Engineered Features Model"
    )

    # 7. Create comparison table
    print("\n7. Creating comparison table...")
    comparison_df = pd.DataFrame({
        'Model': ['Baseline (Original Features)', 'Enhanced (With Engineered Features)'],
        'Features_Count': [len(baseline_cols), len(engineered_cols)],
        'Train_AUC': [baseline_metrics['train_auc'], engineered_metrics['train_auc']],
        'Test_AUC': [baseline_metrics['test_auc'], engineered_metrics['test_auc']],
        'Test_Precision': [baseline_metrics['test_precision'], engineered_metrics['test_precision']],
        'Test_Recall': [baseline_metrics['test_recall'], engineered_metrics['test_recall']],
        'Test_F1': [baseline_metrics['test_f1'], engineered_metrics['test_f1']]
    })

    comparison_df['AUC_Improvement'] = comparison_df['Test_AUC'] - comparison_df['Test_AUC'].iloc[0]
    comparison_df['Meets_Target'] = comparison_df['AUC_Improvement'] >= 0.03

    comparison_df.to_csv('reports/phase2_feature_importance/tables/model_comparison.csv', index=False)
    print("✓ Comparison table saved")

    # 8. Save final feature set
    print("\n8. Saving final feature set...")
    final_features_df = pd.DataFrame({
        'Feature': engineered_cols,
        'Type': ['Original'] * len(baseline_cols) + ['Engineered'] * len(engineered_added)
    })
    final_features_df.to_csv('reports/phase2_feature_importance/tables/final_feature_set.csv', index=False)
    print("✓ Final feature set saved")

    # 9. Create visualizations
    print("\n9. Creating visualizations...")
    create_comparison_visualization(baseline_metrics, engineered_metrics, baseline_proba, engineered_proba, y_test)

    execution_time = time.time() - start_time

    # Calculate improvements
    auc_improvement = engineered_metrics['test_auc'] - baseline_metrics['test_auc']
    meets_target = auc_improvement >= 0.03

    # Build structured output
    output = {
        "success": True,
        "subphase": "Phase 2.3: Feature Engineering Validation",
        "summary": {
            "baseline_features": len(baseline_cols),
            "engineered_features": len(engineered_cols),
            "features_added": len(engineered_added),
            "baseline_auc": float(baseline_metrics['test_auc']),
            "engineered_auc": float(engineered_metrics['test_auc']),
            "auc_improvement": float(auc_improvement),
            "meets_target": bool(meets_target)
        },
        "insights": [
            f"Baseline model ({len(baseline_cols)} features): AUC={baseline_metrics['test_auc']:.4f}",
            f"Enhanced model ({len(engineered_cols)} features): AUC={engineered_metrics['test_auc']:.4f}",
            f"AUC improvement: +{auc_improvement:.4f} ({(auc_improvement/baseline_metrics['test_auc']*100):+.2f}%)",
            f"Target improvement (≥0.03): {'✓ MET' if meets_target else '✗ NOT MET'}",
            f"Added engineered features: {', '.join(engineered_added)}",
            f"Recommended for production: {'Enhanced model with all {len(engineered_cols)} features' if meets_target else 'Further feature engineering needed'}"
        ],
        "outputs": {
            "tables": [
                "model_comparison.csv",
                "final_feature_set.csv"
            ],
            "figures": [
                "feature_validation_comparison.png"
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
    run_feature_validation()
