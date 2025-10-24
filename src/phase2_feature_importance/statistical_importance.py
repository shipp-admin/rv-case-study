"""
Phase 2.1: Statistical Feature Importance
Ranks variables by predictive power using statistical methods

Methods:
1. Mutual Information - Measure mutual dependence
2. ANOVA F-statistic - Test mean differences for numerical variables
3. Chi-Square Test - Independence testing for categorical variables
4. Point-Biserial Correlation - Correlation between binary approval and numerical features
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2
from sklearn.preprocessing import LabelEncoder
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
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def prepare_features(df):
    """Prepare features for statistical analysis"""
    print("\n" + "="*60)
    print("FEATURE PREPARATION")
    print("="*60)

    # Separate numerical and categorical features
    numerical_features = ['FICO_score', 'Loan_Amount', 'Monthly_Gross_Income',
                         'Monthly_Housing_Payment', 'Ever_Bankrupt_or_Foreclose']

    categorical_features = ['Reason', 'Fico_Score_group', 'Employment_Status',
                           'Employment_Sector', 'Lender']

    # Add engineered features if they exist
    if 'DTI' in df.columns:
        numerical_features.extend(['DTI', 'LTI'])
    if 'FICO_Bin_Custom' in df.columns:
        categorical_features.extend(['FICO_Bin_Custom', 'Income_Quartile', 'Loan_Category'])

    print(f"\n✓ Numerical features: {len(numerical_features)}")
    print(f"✓ Categorical features: {len(categorical_features)}")

    # Encode categorical features
    df_encoded = df.copy()
    encoders = {}

    for col in categorical_features:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[f'{col}_encoded'] = le.fit_transform(df_encoded[col].fillna('Unknown'))
            encoders[col] = le

    return df_encoded, numerical_features, categorical_features, encoders


def calculate_mutual_information(df, numerical_features, categorical_features, target='Approved'):
    """
    Calculate mutual information for all features

    Mathematical Foundation:
    Mutual Information measures the reduction in uncertainty about Y (approval)
    given knowledge of X (feature value).

    Formula: MI(X;Y) = Σ P(x,y) log[P(x,y) / (P(x)P(y))]

    Where:
    - P(x,y) = joint probability of X and Y
    - P(x) = marginal probability of X
    - P(y) = marginal probability of Y

    Interpretation:
    - MI = 0: X and Y are independent (feature provides no information)
    - MI > 0: X provides information about Y (higher = more predictive)
    - Captures non-linear relationships unlike correlation
    """
    print("\n" + "="*60)
    print("MUTUAL INFORMATION ANALYSIS")
    print("="*60)

    results = []

    # Numerical features
    # Fill missing values with median to ensure complete data for MI calculation
    X_num = df[numerical_features].fillna(df[numerical_features].median())
    y = df[target]

    # Calculate MI scores using sklearn's mutual_info_classif
    # This uses k-nearest neighbors density estimation for continuous features
    mi_scores = mutual_info_classif(X_num, y, random_state=42)

    for i, feature in enumerate(numerical_features):
        results.append({
            'Feature': feature,
            'Type': 'Numerical',
            'Mutual_Information': mi_scores[i],
            'Rank': 0  # Will be set later after sorting
        })
        print(f"  {feature}: {mi_scores[i]:.4f}")

    # Categorical features
    print(f"\nCategorical Features:")
    for col in categorical_features:
        if col in df.columns:
            encoded_col = f'{col}_encoded'
            if encoded_col in df.columns:
                X_cat = df[[encoded_col]]
                # For categorical features, we use discrete_features=True
                # This tells sklearn to use frequency-based estimation instead of KNN
                mi_score = mutual_info_classif(X_cat, y, discrete_features=True, random_state=42)[0]

                results.append({
                    'Feature': col,
                    'Type': 'Categorical',
                    'Mutual_Information': mi_score,
                    'Rank': 0
                })
                print(f"  {col}: {mi_score:.4f}")

    # Sort by MI score (descending) and assign ranks
    # Higher MI = more informative feature
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Mutual_Information', ascending=False).reset_index(drop=True)
    results_df['Rank'] = range(1, len(results_df) + 1)

    return results_df


def calculate_anova_f_statistics(df, numerical_features, target='Approved'):
    """
    Calculate ANOVA F-statistics for numerical features

    Mathematical Foundation:
    ANOVA (Analysis of Variance) tests if the means of a numerical variable
    differ significantly between approved and denied groups.

    Formula: F = MS_between / MS_within

    Where:
    - MS_between = Between-group variance = SS_between / df_between
    - MS_within = Within-group variance = SS_within / df_within
    - SS_between = Σ n_i (mean_i - grand_mean)²
    - SS_within = Σ Σ (x_ij - mean_i)²

    Interpretation:
    - F close to 1: Group means are similar (feature not predictive)
    - F >> 1: Group means differ significantly (feature is predictive)
    - p-value < 0.05: Reject null hypothesis (means are different)
    - Higher F-statistic = stronger discriminative power
    """
    print("\n" + "="*60)
    print("ANOVA F-STATISTIC ANALYSIS")
    print("="*60)

    results = []

    # Fill missing values with median for complete data
    X = df[numerical_features].fillna(df[numerical_features].median())
    y = df[target]

    # Calculate F-statistics using sklearn's f_classif
    # This computes one-way ANOVA for each feature vs target
    f_scores, p_values = f_classif(X, y)

    for i, feature in enumerate(numerical_features):
        results.append({
            'Feature': feature,
            'F_Statistic': f_scores[i],
            'P_Value': p_values[i],
            'Significant': 'Yes' if p_values[i] < 0.05 else 'No'
        })
        print(f"  {feature}:")
        print(f"    F-statistic: {f_scores[i]:.2f}")
        print(f"    p-value: {p_values[i]:.4e}")
        print(f"    Significant: {'Yes' if p_values[i] < 0.05 else 'No'}")

    return pd.DataFrame(results)


def calculate_chi_square(df, categorical_features, target='Approved'):
    """
    Calculate chi-square statistics for categorical features

    Mathematical Foundation:
    Chi-square test assesses independence between two categorical variables.
    Tests if the distribution of approval/denial differs across feature categories.

    Formula: χ² = Σ [(Observed - Expected)² / Expected]

    Where:
    - Observed = actual counts in each cell of contingency table
    - Expected = (row_total × column_total) / grand_total
    - Sum is over all cells in the contingency table

    Interpretation:
    - χ² = 0: Variables are perfectly independent (feature not predictive)
    - χ² >> 0: Variables are dependent (feature is predictive)
    - p-value < 0.05: Reject independence hypothesis
    - Higher χ² = stronger association with approval outcome
    """
    print("\n" + "="*60)
    print("CHI-SQUARE TEST ANALYSIS")
    print("="*60)

    results = []

    for col in categorical_features:
        if col in df.columns:
            # Create contingency table (cross-tabulation)
            # Rows = feature categories, Columns = Approved/Denied
            contingency = pd.crosstab(df[col].fillna('Unknown'), df[target])

            # Chi-square test of independence
            # Returns: χ² statistic, p-value, degrees of freedom, expected frequencies
            chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)

            results.append({
                'Feature': col,
                'Chi2_Statistic': chi2_stat,
                'P_Value': p_value,
                'DOF': dof,
                'Significant': 'Yes' if p_value < 0.05 else 'No'
            })

            print(f"\n  {col}:")
            print(f"    Chi2-statistic: {chi2_stat:.2f}")
            print(f"    p-value: {p_value:.4e}")
            print(f"    Degrees of freedom: {dof}")
            print(f"    Significant: {'Yes' if p_value < 0.05 else 'No'}")

    return pd.DataFrame(results)


def calculate_point_biserial_correlation(df, numerical_features, target='Approved'):
    """
    Calculate point-biserial correlation for numerical features

    Mathematical Foundation:
    Point-biserial correlation measures the strength of association between
    a continuous variable (feature) and a binary variable (approval).

    Formula: r_pb = (M₁ - M₀) / S × √[n₁n₀ / n(n-1)]

    Where:
    - M₁ = mean of continuous variable when binary variable = 1 (approved)
    - M₀ = mean of continuous variable when binary variable = 0 (denied)
    - S = overall standard deviation of continuous variable
    - n₁ = count when binary variable = 1
    - n₀ = count when binary variable = 0
    - n = total count (n₁ + n₀)

    Interpretation:
    - r_pb ranges from -1 to +1
    - r_pb > 0: Higher feature values → Higher approval probability
    - r_pb < 0: Higher feature values → Lower approval probability
    - |r_pb| close to 0: Weak linear relationship
    - |r_pb| > 0.3: Moderate to strong linear relationship
    """
    print("\n" + "="*60)
    print("POINT-BISERIAL CORRELATION ANALYSIS")
    print("="*60)

    results = []

    for feature in numerical_features:
        # Remove NaN values to ensure valid correlation calculation
        valid_data = df[[feature, target]].dropna()

        # Calculate point-biserial correlation coefficient
        # Returns: correlation coefficient and two-tailed p-value
        correlation, p_value = stats.pointbiserialr(valid_data[target], valid_data[feature])

        results.append({
            'Feature': feature,
            'Correlation': correlation,
            'P_Value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })

        print(f"  {feature}:")
        print(f"    Correlation: {correlation:.4f}")
        print(f"    p-value: {p_value:.4e}")
        print(f"    Significant: {'Yes' if p_value < 0.05 else 'No'}")

    return pd.DataFrame(results)


def create_feature_importance_visualizations(mi_df, anova_df, chi2_df, pb_df):
    """Create comprehensive feature importance visualizations"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Mutual Information (top 10)
    ax = axes[0, 0]
    top_mi = mi_df.head(10)
    ax.barh(range(len(top_mi)), top_mi['Mutual_Information'], color='steelblue')
    ax.set_yticks(range(len(top_mi)))
    ax.set_yticklabels(top_mi['Feature'])
    ax.set_xlabel('Mutual Information Score')
    ax.set_title('Top 10 Features by Mutual Information', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 2. ANOVA F-Statistics
    ax = axes[0, 1]
    anova_sorted = anova_df.sort_values('F_Statistic', ascending=False)
    colors = ['green' if sig == 'Yes' else 'gray' for sig in anova_sorted['Significant']]
    ax.barh(range(len(anova_sorted)), anova_sorted['F_Statistic'], color=colors)
    ax.set_yticks(range(len(anova_sorted)))
    ax.set_yticklabels(anova_sorted['Feature'])
    ax.set_xlabel('F-Statistic')
    ax.set_title('ANOVA F-Statistics (Numerical Features)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 3. Chi-Square Statistics
    ax = axes[1, 0]
    chi2_sorted = chi2_df.sort_values('Chi2_Statistic', ascending=False).head(10)
    colors = ['green' if sig == 'Yes' else 'gray' for sig in chi2_sorted['Significant']]
    ax.barh(range(len(chi2_sorted)), chi2_sorted['Chi2_Statistic'], color=colors)
    ax.set_yticks(range(len(chi2_sorted)))
    ax.set_yticklabels(chi2_sorted['Feature'])
    ax.set_xlabel('Chi-Square Statistic')
    ax.set_title('Top 10 Chi-Square Statistics (Categorical Features)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    # 4. Point-Biserial Correlation (absolute values)
    ax = axes[1, 1]
    pb_df['Abs_Correlation'] = pb_df['Correlation'].abs()
    pb_sorted = pb_df.sort_values('Abs_Correlation', ascending=False)
    colors = ['green' if sig == 'Yes' else 'gray' for sig in pb_sorted['Significant']]
    ax.barh(range(len(pb_sorted)), pb_sorted['Abs_Correlation'], color=colors)
    ax.set_yticks(range(len(pb_sorted)))
    ax.set_yticklabels(pb_sorted['Feature'])
    ax.set_xlabel('|Point-Biserial Correlation|')
    ax.set_title('Point-Biserial Correlations (Numerical Features)', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/phase2_feature_importance/figures/statistical_importance.png', dpi=300, bbox_inches='tight')
    plt.close()


def identify_features_to_drop(mi_df, anova_df, chi2_df):
    """Identify features that can be dropped (low importance + not significant)"""
    print("\n" + "="*60)
    print("FEATURES TO DROP ANALYSIS")
    print("="*60)

    features_to_drop = []

    # Features with very low mutual information (< 0.01)
    low_mi = mi_df[mi_df['Mutual_Information'] < 0.01]['Feature'].tolist()

    # Non-significant numerical features
    non_sig_anova = anova_df[anova_df['P_Value'] > 0.05]['Feature'].tolist()

    # Non-significant categorical features
    non_sig_chi2 = chi2_df[chi2_df['P_Value'] > 0.05]['Feature'].tolist()

    # Combine
    all_candidates = set(low_mi + non_sig_anova + non_sig_chi2)

    for feature in all_candidates:
        # Check if consistently low across metrics
        in_low_mi = feature in low_mi
        in_non_sig = feature in (non_sig_anova + non_sig_chi2)

        if in_low_mi and in_non_sig:
            features_to_drop.append(feature)
            print(f"  ❌ {feature}: Low MI + Not significant")
        elif in_non_sig:
            print(f"  ⚠️  {feature}: Not significant (p > 0.05)")

    print(f"\n📋 Recommended features to drop: {len(features_to_drop)}")
    print(f"   {features_to_drop if features_to_drop else 'None - all features show some predictive power'}")

    return features_to_drop


def run_statistical_importance():
    """Main function to run statistical feature importance analysis"""
    start_time = time.time()

    print("=" * 70)
    print("Phase 2.1: Statistical Feature Importance")
    print("=" * 70)

    # Create output directories
    create_output_dirs()
    print("✓ Output directories created")

    # Load data
    print("\n1. Loading data...")
    df, report = load_and_validate()
    print(f"✓ Data loaded: {len(df):,} rows, {len(df.columns)} columns")

    # Try to load engineered features from Phase 1.2
    try:
        df_engineered = pd.read_csv('data/processed/features_engineered.csv')
        # Merge engineered features
        engineered_cols = ['DTI', 'LTI', 'FICO_Bin_Custom', 'Income_Quartile', 'Loan_Category']
        for col in engineered_cols:
            if col in df_engineered.columns:
                df[col] = df_engineered[col]
        print(f"✓ Loaded {len([c for c in engineered_cols if c in df.columns])} engineered features from Phase 1.2")
    except:
        print("⚠️  No engineered features found from Phase 1.2 (run Phase 1.2 first for full analysis)")

    # Prepare features
    print("\n2. Preparing features...")
    df_encoded, numerical_features, categorical_features, encoders = prepare_features(df)
    print("✓ Features prepared and encoded")

    # Calculate mutual information
    print("\n3. Calculating mutual information...")
    mi_df = calculate_mutual_information(df_encoded, numerical_features, categorical_features)
    print("✓ Mutual information calculated")

    # Calculate ANOVA F-statistics
    print("\n4. Calculating ANOVA F-statistics...")
    anova_df = calculate_anova_f_statistics(df_encoded, numerical_features)
    print("✓ ANOVA F-statistics calculated")

    # Calculate chi-square statistics
    print("\n5. Calculating chi-square statistics...")
    chi2_df = calculate_chi_square(df_encoded, categorical_features)
    print("✓ Chi-square statistics calculated")

    # Calculate point-biserial correlations
    print("\n6. Calculating point-biserial correlations...")
    pb_df = calculate_point_biserial_correlation(df_encoded, numerical_features)
    print("✓ Point-biserial correlations calculated")

    # Identify features to drop
    print("\n7. Identifying features to drop...")
    features_to_drop = identify_features_to_drop(mi_df, anova_df, chi2_df)
    print("✓ Feature drop analysis complete")

    # Create visualizations
    print("\n8. Creating visualizations...")
    create_feature_importance_visualizations(mi_df, anova_df, chi2_df, pb_df)
    print("✓ Visualizations saved")

    # Save results
    print("\n9. Saving results...")
    mi_df.to_csv('reports/phase2_feature_importance/tables/mutual_information.csv', index=False)
    anova_df.to_csv('reports/phase2_feature_importance/tables/anova_f_statistics.csv', index=False)
    chi2_df.to_csv('reports/phase2_feature_importance/tables/chi_square_tests.csv', index=False)
    pb_df.to_csv('reports/phase2_feature_importance/tables/point_biserial_correlations.csv', index=False)

    # Save features to drop
    pd.DataFrame({'Feature': features_to_drop}).to_csv(
        'reports/phase2_feature_importance/tables/features_to_drop.csv', index=False
    )
    print("✓ Results saved")

    execution_time = time.time() - start_time

    # Build structured output
    output = {
        "success": True,
        "subphase": "Phase 2.1: Statistical Feature Importance",
        "summary": {
            "total_features": len(numerical_features) + len(categorical_features),
            "numerical_features": len(numerical_features),
            "categorical_features": len(categorical_features),
            "features_to_drop": len(features_to_drop),
            "top_feature": mi_df.iloc[0]['Feature'],
            "top_mi_score": float(mi_df.iloc[0]['Mutual_Information'])
        },
        "insights": [
            f"Analyzed {len(numerical_features) + len(categorical_features)} features using 4 statistical methods",
            f"Top feature by mutual information: {mi_df.iloc[0]['Feature']} (MI={mi_df.iloc[0]['Mutual_Information']:.4f})",
            f"{len(anova_df[anova_df['Significant']=='Yes'])} numerical features are statistically significant (p<0.05)",
            f"{len(chi2_df[chi2_df['Significant']=='Yes'])} categorical features are statistically significant (p<0.05)",
            f"Recommended {len(features_to_drop)} features for removal (low importance + not significant)" if features_to_drop else "All features show predictive power - none recommended for removal",
        ],
        "outputs": {
            "tables": [
                "mutual_information.csv",
                "anova_f_statistics.csv",
                "chi_square_tests.csv",
                "point_biserial_correlations.csv",
                "features_to_drop.csv"
            ],
            "figures": [
                "statistical_importance.png"
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
    run_statistical_importance()
