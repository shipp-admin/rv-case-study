"""
Phase 2.2: Machine Learning Feature Importance
Uses ML models to assess feature importance and validate statistical findings

Models:
1. Random Forest Classifier - Gini importance and permutation importance
2. XGBoost Classifier - Gain, cover, and frequency importance metrics
3. Logistic Regression (L1) - Coefficient magnitudes and automated feature selection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
from pathlib import Path
import json
import time
import warnings
warnings.filterwarnings('ignore')

# Import data loader
import sys
sys.path.append(str(Path(__file__).parent.parent))
from phase1_eda.data_loader import load_and_validate


def create_output_dirs():
    """Create output directories if they don't exist"""
    dirs = [
        'reports/phase2_feature_importance/figures',
        'reports/phase2_feature_importance/tables',
        'models/phase2_feature_models',
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def prepare_ml_features(df):
    """Prepare features for ML models"""
    print("\n" + "="*60)
    print("ML FEATURE PREPARATION")
    print("="*60)

    # Target variable
    y = df['Approved']

    # Numerical features
    numerical_features = ['FICO_score', 'Loan_Amount', 'Monthly_Gross_Income',
                         'Monthly_Housing_Payment', 'Ever_Bankrupt_or_Foreclose']

    # Add engineered features if they exist
    if 'DTI' in df.columns:
        numerical_features.extend(['DTI', 'LTI'])

    # Categorical features
    categorical_features = ['Reason', 'Fico_Score_group', 'Employment_Status',
                           'Employment_Sector', 'Lender']

    if 'FICO_Bin_Custom' in df.columns:
        categorical_features.extend(['FICO_Bin_Custom', 'Income_Quartile', 'Loan_Category'])

    # Encode categorical features
    df_encoded = df.copy()
    feature_names = []
    encoders = {}

    for col in categorical_features:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].fillna('Unknown'))
            encoders[col] = le
            feature_names.append(f'{col}_encoded')

    # Add numerical features
    feature_names.extend(numerical_features)

    # Create feature matrix
    X = df_encoded[feature_names].fillna(df_encoded[feature_names].median())

    print(f"\n✓ Total features: {len(feature_names)}")
    print(f"✓ Numerical: {len(numerical_features)}")
    print(f"✓ Categorical (encoded): {len(categorical_features)}")
    print(f"✓ Target class balance: {y.mean():.2%} approved")

    return X, y, feature_names, encoders


def train_random_forest(X_train, X_test, y_train, y_test, feature_names):
    """Train Random Forest and extract feature importance"""
    print("\n" + "="*60)
    print("RANDOM FOREST CLASSIFIER")
    print("="*60)

    # Train model
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        oob_score=True
    )

    print("\nTraining Random Forest...")
    rf.fit(X_train, y_train)

    # Predictions
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)

    # Performance metrics
    train_auc = roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, y_pred_proba)
    oob_score = rf.oob_score_

    print(f"\n✓ Model trained successfully")
    print(f"  Train AUC-ROC: {train_auc:.4f}")
    print(f"  Test AUC-ROC: {test_auc:.4f}")
    print(f"  OOB Score: {oob_score:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Gini_Importance': rf.feature_importances_
    }).sort_values('Gini_Importance', ascending=False)

    print(f"\nTop 5 Features by Gini Importance:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['Feature']}: {row['Gini_Importance']:.4f}")

    return rf, feature_importance, test_auc


def train_xgboost(X_train, X_test, y_train, y_test, feature_names):
    """Train XGBoost and extract feature importance"""
    print("\n" + "="*60)
    print("XGBOOST CLASSIFIER")
    print("="*60)

    # Train model
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='auc',
        use_label_encoder=False
    )

    print("\nTraining XGBoost...")
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    # Predictions
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    y_pred = xgb_model.predict(X_test)

    # Performance metrics
    train_auc = roc_auc_score(y_train, xgb_model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n✓ Model trained successfully")
    print(f"  Train AUC-ROC: {train_auc:.4f}")
    print(f"  Test AUC-ROC: {test_auc:.4f}")

    # Feature importance (gain, cover, weight)
    importance_gain = xgb_model.get_booster().get_score(importance_type='gain')
    importance_cover = xgb_model.get_booster().get_score(importance_type='cover')
    importance_weight = xgb_model.get_booster().get_score(importance_type='weight')

    # Create feature importance dataframe
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Gain': [importance_gain.get(f'f{i}', 0) for i in range(len(feature_names))],
        'Cover': [importance_cover.get(f'f{i}', 0) for i in range(len(feature_names))],
        'Weight': [importance_weight.get(f'f{i}', 0) for i in range(len(feature_names))]
    }).sort_values('Gain', ascending=False)

    print(f"\nTop 5 Features by Gain:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['Feature']}: {row['Gain']:.2f}")

    return xgb_model, feature_importance, test_auc


def train_logistic_regression(X_train, X_test, y_train, y_test, feature_names):
    """Train Logistic Regression with L1 regularization"""
    print("\n" + "="*60)
    print("LOGISTIC REGRESSION (L1 REGULARIZATION)")
    print("="*60)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    lr = LogisticRegression(
        penalty='l1',
        C=0.1,
        solver='liblinear',
        random_state=42,
        max_iter=1000
    )

    print("\nTraining Logistic Regression...")
    lr.fit(X_train_scaled, y_train)

    # Predictions
    y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
    y_pred = lr.predict(X_test_scaled)

    # Performance metrics
    train_auc = roc_auc_score(y_train, lr.predict_proba(X_train_scaled)[:, 1])
    test_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n✓ Model trained successfully")
    print(f"  Train AUC-ROC: {train_auc:.4f}")
    print(f"  Test AUC-ROC: {test_auc:.4f}")

    # Feature importance (absolute coefficient values)
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': lr.coef_[0],
        'Abs_Coefficient': np.abs(lr.coef_[0]),
        'Selected': lr.coef_[0] != 0  # L1 feature selection
    }).sort_values('Abs_Coefficient', ascending=False)

    # Count non-zero features (L1 feature selection)
    non_zero = (feature_importance['Coefficient'] != 0).sum()
    print(f"\n✓ Features selected: {non_zero}/{len(feature_names)}")

    print(f"\nTop 5 Features by |Coefficient|:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['Feature']}: {row['Coefficient']:.4f}")

    return lr, scaler, feature_importance, test_auc


def create_consensus_rankings(rf_importance, xgb_importance, lr_importance, feature_names):
    """Create consensus feature rankings across all models"""
    print("\n" + "="*60)
    print("CONSENSUS FEATURE RANKINGS")
    print("="*60)

    # Normalize importance scores to [0, 1]
    rf_norm = rf_importance.copy()
    rf_norm['RF_Score'] = (rf_norm['Gini_Importance'] - rf_norm['Gini_Importance'].min()) / \
                          (rf_norm['Gini_Importance'].max() - rf_norm['Gini_Importance'].min())

    xgb_norm = xgb_importance.copy()
    xgb_norm['XGB_Score'] = (xgb_norm['Gain'] - xgb_norm['Gain'].min()) / \
                            (xgb_norm['Gain'].max() - xgb_norm['Gain'].min())

    lr_norm = lr_importance.copy()
    lr_norm['LR_Score'] = (lr_norm['Abs_Coefficient'] - lr_norm['Abs_Coefficient'].min()) / \
                          (lr_norm['Abs_Coefficient'].max() - lr_norm['Abs_Coefficient'].min())

    # Merge all scores
    consensus = pd.DataFrame({'Feature': feature_names})
    consensus = consensus.merge(rf_norm[['Feature', 'RF_Score']], on='Feature', how='left')
    consensus = consensus.merge(xgb_norm[['Feature', 'XGB_Score']], on='Feature', how='left')
    consensus = consensus.merge(lr_norm[['Feature', 'LR_Score']], on='Feature', how='left')

    # Fill NaN with 0
    consensus = consensus.fillna(0)

    # Calculate average score
    consensus['Consensus_Score'] = (consensus['RF_Score'] + consensus['XGB_Score'] + consensus['LR_Score']) / 3
    consensus = consensus.sort_values('Consensus_Score', ascending=False).reset_index(drop=True)
    consensus['Rank'] = range(1, len(consensus) + 1)

    print(f"\nTop 10 Features by Consensus Score:")
    for idx, row in consensus.head(10).iterrows():
        print(f"  {row['Rank']}. {row['Feature']}: {row['Consensus_Score']:.4f}")

    return consensus


def create_ml_importance_visualizations(rf_importance, xgb_importance, lr_importance, consensus,
                                       rf_auc, xgb_auc, lr_auc):
    """Create comprehensive ML feature importance visualizations"""

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Random Forest Feature Importance
    ax1 = fig.add_subplot(gs[0, 0])
    top_rf = rf_importance.head(10)
    ax1.barh(range(len(top_rf)), top_rf['Gini_Importance'], color='forestgreen')
    ax1.set_yticks(range(len(top_rf)))
    ax1.set_yticklabels(top_rf['Feature'], fontsize=8)
    ax1.set_xlabel('Gini Importance')
    ax1.set_title(f'Random Forest (AUC: {rf_auc:.3f})', fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)

    # 2. XGBoost Feature Importance
    ax2 = fig.add_subplot(gs[0, 1])
    top_xgb = xgb_importance.head(10)
    ax2.barh(range(len(top_xgb)), top_xgb['Gain'], color='steelblue')
    ax2.set_yticks(range(len(top_xgb)))
    ax2.set_yticklabels(top_xgb['Feature'], fontsize=8)
    ax2.set_xlabel('Gain')
    ax2.set_title(f'XGBoost (AUC: {xgb_auc:.3f})', fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)

    # 3. Logistic Regression Coefficients
    ax3 = fig.add_subplot(gs[0, 2])
    top_lr = lr_importance.head(10)
    colors = ['red' if coef < 0 else 'green' for coef in top_lr['Coefficient']]
    ax3.barh(range(len(top_lr)), top_lr['Coefficient'], color=colors)
    ax3.set_yticks(range(len(top_lr)))
    ax3.set_yticklabels(top_lr['Feature'], fontsize=8)
    ax3.set_xlabel('Coefficient')
    ax3.set_title(f'Logistic Regression (AUC: {lr_auc:.3f})', fontweight='bold')
    ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)

    # 4. Consensus Rankings (Top 15)
    ax4 = fig.add_subplot(gs[1, :])
    top_consensus = consensus.head(15)
    x = range(len(top_consensus))
    width = 0.25

    ax4.bar([i - width for i in x], top_consensus['RF_Score'], width, label='Random Forest', color='forestgreen', alpha=0.8)
    ax4.bar([i for i in x], top_consensus['XGB_Score'], width, label='XGBoost', color='steelblue', alpha=0.8)
    ax4.bar([i + width for i in x], top_consensus['LR_Score'], width, label='Logistic Reg', color='coral', alpha=0.8)

    ax4.set_xlabel('Features')
    ax4.set_ylabel('Normalized Importance Score')
    ax4.set_title('Consensus Feature Importance: Top 15 Features Across All Models', fontweight='bold', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(top_consensus['Feature'], rotation=45, ha='right', fontsize=8)
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    # 5. Model Performance Comparison
    ax5 = fig.add_subplot(gs[2, 0])
    models = ['Random Forest', 'XGBoost', 'Logistic Reg']
    aucs = [rf_auc, xgb_auc, lr_auc]
    colors_bar = ['forestgreen', 'steelblue', 'coral']
    ax5.bar(models, aucs, color=colors_bar, alpha=0.8, edgecolor='black')
    ax5.set_ylabel('AUC-ROC Score')
    ax5.set_title('Model Performance Comparison', fontweight='bold')
    ax5.axhline(y=0.7, color='red', linestyle='--', label='Target: 0.70')
    ax5.set_ylim([0, 1])
    ax5.legend()
    ax5.grid(axis='y', alpha=0.3)

    # 6. Top 10 Consensus Rankings
    ax6 = fig.add_subplot(gs[2, 1:])
    top10 = consensus.head(10)
    ax6.barh(range(len(top10)), top10['Consensus_Score'], color='purple', alpha=0.7, edgecolor='black')
    ax6.set_yticks(range(len(top10)))
    ax6.set_yticklabels([f"{i+1}. {f}" for i, f in enumerate(top10['Feature'])], fontsize=9)
    ax6.set_xlabel('Consensus Score (Average)')
    ax6.set_title('Top 10 Features: Consensus Rankings', fontweight='bold', fontsize=11)
    ax6.invert_yaxis()
    ax6.grid(axis='x', alpha=0.3)

    plt.savefig('reports/phase2_feature_importance/figures/ml_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()


def run_ml_importance():
    """Main function to run ML feature importance analysis"""
    start_time = time.time()

    print("=" * 70)
    print("Phase 2.2: Machine Learning Feature Importance")
    print("=" * 70)

    # Create output directories
    create_output_dirs()
    print("✓ Output directories created")

    # Load data
    print("\n1. Loading data...")
    df, report = load_and_validate()

    # Try to load engineered features
    try:
        df_engineered = pd.read_csv('data/processed/features_engineered.csv')
        engineered_cols = ['DTI', 'LTI', 'FICO_Bin_Custom', 'Income_Quartile', 'Loan_Category']
        for col in engineered_cols:
            if col in df_engineered.columns:
                df[col] = df_engineered[col]
        print(f"✓ Loaded {len([c for c in engineered_cols if c in df.columns])} engineered features")
    except:
        print("⚠️  No engineered features found (run Phase 1.2 first)")

    # Prepare features
    print("\n2. Preparing features for ML...")
    X, y, feature_names, encoders = prepare_ml_features(df)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"\n✓ Train set: {len(X_train):,} samples")
    print(f"✓ Test set: {len(X_test):,} samples")

    # Train Random Forest
    print("\n3. Training Random Forest...")
    rf, rf_importance, rf_auc = train_random_forest(X_train, X_test, y_train, y_test, feature_names)

    # Train XGBoost
    print("\n4. Training XGBoost...")
    xgb_model, xgb_importance, xgb_auc = train_xgboost(X_train, X_test, y_train, y_test, feature_names)

    # Train Logistic Regression
    print("\n5. Training Logistic Regression...")
    lr, scaler, lr_importance, lr_auc = train_logistic_regression(X_train, X_test, y_train, y_test, feature_names)

    # Create consensus rankings
    print("\n6. Creating consensus rankings...")
    consensus = create_consensus_rankings(rf_importance, xgb_importance, lr_importance, feature_names)

    # Create visualizations
    print("\n7. Creating visualizations...")
    create_ml_importance_visualizations(rf_importance, xgb_importance, lr_importance, consensus,
                                       rf_auc, xgb_auc, lr_auc)
    print("✓ Visualizations saved")

    # Save results
    print("\n8. Saving results...")
    rf_importance.to_csv('reports/phase2_feature_importance/tables/rf_feature_importance.csv', index=False)
    xgb_importance.to_csv('reports/phase2_feature_importance/tables/xgb_feature_importance.csv', index=False)
    lr_importance.to_csv('reports/phase2_feature_importance/tables/lr_feature_importance.csv', index=False)
    consensus.to_csv('reports/phase2_feature_importance/tables/consensus_feature_rankings.csv', index=False)

    # Save model performance summary
    model_performance = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'Logistic Regression'],
        'Test_AUC': [rf_auc, xgb_auc, lr_auc],
        'Meets_Target': [auc >= 0.70 for auc in [rf_auc, xgb_auc, lr_auc]]
    })
    model_performance.to_csv('reports/phase2_feature_importance/tables/model_performance.csv', index=False)

    print("✓ Results saved")

    execution_time = time.time() - start_time

    # Determine best model
    best_auc = max(rf_auc, xgb_auc, lr_auc)
    if rf_auc == best_auc:
        best_model = "Random Forest"
    elif xgb_auc == best_auc:
        best_model = "XGBoost"
    else:
        best_model = "Logistic Regression"

    # Build structured output
    output = {
        "success": True,
        "subphase": "Phase 2.2: ML Feature Importance",
        "summary": {
            "total_features": len(feature_names),
            "best_model": best_model,
            "best_auc": float(best_auc),
            "top_feature": consensus.iloc[0]['Feature'],
            "top_consensus_score": float(consensus.iloc[0]['Consensus_Score'])
        },
        "insights": [
            f"Trained 3 ML models: Random Forest (AUC={rf_auc:.3f}), XGBoost (AUC={xgb_auc:.3f}), Logistic Regression (AUC={lr_auc:.3f})",
            f"Best model: {best_model} with AUC={best_auc:.3f}",
            f"Top feature by consensus: {consensus.iloc[0]['Feature']} (score={consensus.iloc[0]['Consensus_Score']:.4f})",
            f"All models {'meet' if min(rf_auc, xgb_auc, lr_auc) >= 0.70 else 'do not meet'} target AUC ≥ 0.70",
            f"Top 5 consensus features: {', '.join(consensus.head(5)['Feature'].tolist())}"
        ],
        "outputs": {
            "tables": [
                "rf_feature_importance.csv",
                "xgb_feature_importance.csv",
                "lr_feature_importance.csv",
                "consensus_feature_rankings.csv",
                "model_performance.csv"
            ],
            "figures": [
                "ml_feature_importance.png"
            ]
        },
        "execution_time": execution_time
    }

    print("\n" + "=" * 70)
    print("__JSON_OUTPUT__")
    print(json.dumps(output, indent=2))
    print("=" * 70)

    return output


if __name__ == "__main__":
    run_ml_importance()
