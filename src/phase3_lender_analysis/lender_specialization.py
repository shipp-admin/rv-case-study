"""
Phase 3.3: Lender Specialization Analysis
Identifies clear differences in customer types each lender approves through:
- ANOVA testing for numerical variable differences
- Chi-square testing for categorical variable differences
- K-means clustering to identify customer segments
- Lender preference analysis across segments
"""

import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import f_oneway, chi2_contingency
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from phase1_eda.data_loader import load_and_validate

# Setup paths
REPORTS_DIR = Path(__file__).parent.parent.parent / 'reports' / 'phase3_lender_analysis'
TABLES_DIR = REPORTS_DIR / 'tables'
FIGURES_DIR = REPORTS_DIR / 'figures'

# Create directories
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def prepare_features(df):
    """Prepare engineered features for analysis"""
    df = df.copy()

    # DTI and LTI ratios
    df['DTI'] = df['Monthly_Housing_Payment'] / df['Monthly_Gross_Income']
    df['LTI'] = df['Loan_Amount'] / (df['Monthly_Gross_Income'] * 12)

    # Income quartiles
    df['Income_Quartile'] = pd.qcut(df['Monthly_Gross_Income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])

    # Loan brackets
    df['Loan_Bracket'] = pd.cut(
        df['Loan_Amount'],
        bins=[0, 30000, 60000, float('inf')],
        labels=['Small', 'Medium', 'Large']
    )

    return df

def perform_anova_tests(df):
    """
    ANOVA: Test if mean FICO, income, loan amounts differ across lenders for approvals
    """
    print("\n" + "="*80)
    print("ANOVA TESTS - Numerical Variable Differences Across Lenders")
    print("="*80)

    # Filter to approved applications only
    approved = df[df['Approved'] == 1].copy()

    # Group by lender
    lender_a = approved[approved['Lender'] == 'A']
    lender_b = approved[approved['Lender'] == 'B']
    lender_c = approved[approved['Lender'] == 'C']

    anova_results = []

    numerical_vars = ['FICO_score', 'Monthly_Gross_Income', 'Loan_Amount', 'DTI', 'LTI']

    for var in numerical_vars:
        # Get values per lender
        a_vals = lender_a[var].dropna()
        b_vals = lender_b[var].dropna()
        c_vals = lender_c[var].dropna()

        # ANOVA test
        f_stat, p_value = f_oneway(a_vals, b_vals, c_vals)

        # Calculate means
        mean_a = a_vals.mean()
        mean_b = b_vals.mean()
        mean_c = c_vals.mean()

        significant = p_value < 0.05

        anova_results.append({
            'Variable': var,
            'F_Statistic': f_stat,
            'P_Value': p_value,
            'Significant': significant,
            'Mean_A': mean_a,
            'Mean_B': mean_b,
            'Mean_C': mean_c,
            'Overall_Mean': approved[var].mean()
        })

        print(f"\n{var}:")
        print(f"  F-statistic: {f_stat:.4f}, p-value: {p_value:.6f} {'✓ Significant' if significant else 'Not significant'}")
        print(f"  Lender A: {mean_a:.2f}")
        print(f"  Lender B: {mean_b:.2f}")
        print(f"  Lender C: {mean_c:.2f}")

    anova_df = pd.DataFrame(anova_results)
    anova_df.to_csv(TABLES_DIR / 'anova_results.csv', index=False)
    print(f"\n✓ ANOVA results saved to {TABLES_DIR / 'anova_results.csv'}")

    return anova_df

def perform_chi_square_tests(df):
    """
    Chi-Square: Test if approval rates differ by sector, reason, employment status
    """
    print("\n" + "="*80)
    print("CHI-SQUARE TESTS - Categorical Variable Differences Across Lenders")
    print("="*80)

    # Filter to approved applications
    approved = df[df['Approved'] == 1].copy()

    chi_square_results = []

    categorical_vars = ['Reason', 'Employment_Status', 'Income_Quartile', 'Loan_Bracket']

    for var in categorical_vars:
        # Create contingency table: Lender × Variable
        contingency = pd.crosstab(approved['Lender'], approved[var])

        # Chi-square test
        chi2, p_value, dof, expected = chi2_contingency(contingency)

        significant = p_value < 0.05

        chi_square_results.append({
            'Variable': var,
            'Chi2_Statistic': chi2,
            'P_Value': p_value,
            'Degrees_of_Freedom': dof,
            'Significant': significant
        })

        print(f"\n{var}:")
        print(f"  Chi2-statistic: {chi2:.4f}, p-value: {p_value:.6f} {'✓ Significant' if significant else 'Not significant'}")
        print(f"  Degrees of freedom: {dof}")

        # Show distribution percentages
        print(f"  Distribution by Lender:")
        for lender in ['A', 'B', 'C']:
            lender_dist = contingency.loc[lender] / contingency.loc[lender].sum()
            print(f"    Lender {lender}: {lender_dist.to_dict()}")

    chi_df = pd.DataFrame(chi_square_results)
    chi_df.to_csv(TABLES_DIR / 'chi_square_results.csv', index=False)
    print(f"\n✓ Chi-square results saved to {TABLES_DIR / 'chi_square_results.csv'}")

    return chi_df

def perform_clustering(df):
    """
    K-means clustering on approved customers to identify segments
    """
    print("\n" + "="*80)
    print("K-MEANS CLUSTERING - Customer Segmentation")
    print("="*80)

    # Filter to approved applications
    approved = df[df['Approved'] == 1].copy()

    # Features for clustering: FICO, income, loan amount
    clustering_features = ['FICO_score', 'Monthly_Gross_Income', 'Loan_Amount']
    X = approved[clustering_features].dropna()

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine optimal k (test 3-6 clusters)
    silhouette_scores = []
    k_range = range(3, 7)

    print("\nDetermining optimal number of clusters...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        silhouette_scores.append(score)
        print(f"  k={k}: silhouette score = {score:.4f}")

    # Choose k with highest silhouette score
    optimal_k = k_range[np.argmax(silhouette_scores)]
    print(f"\n✓ Optimal k = {optimal_k} clusters")

    # Final clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    approved_with_cluster = approved.loc[X.index].copy()
    approved_with_cluster['Cluster'] = kmeans.fit_predict(X_scaled)

    # Analyze cluster characteristics
    cluster_profiles = []
    for cluster_id in range(optimal_k):
        cluster_data = approved_with_cluster[approved_with_cluster['Cluster'] == cluster_id]

        profile = {
            'Cluster': cluster_id,
            'Size': len(cluster_data),
            'Pct_of_Approvals': len(cluster_data) / len(approved_with_cluster) * 100,
            'Mean_FICO': cluster_data['FICO_score'].mean(),
            'Mean_Income': cluster_data['Monthly_Gross_Income'].mean(),
            'Mean_Loan': cluster_data['Loan_Amount'].mean(),
            'Mean_DTI': cluster_data['DTI'].mean(),
            'Lender_A_Pct': (cluster_data['Lender'] == 'A').sum() / len(cluster_data) * 100,
            'Lender_B_Pct': (cluster_data['Lender'] == 'B').sum() / len(cluster_data) * 100,
            'Lender_C_Pct': (cluster_data['Lender'] == 'C').sum() / len(cluster_data) * 100
        }
        cluster_profiles.append(profile)

        print(f"\nCluster {cluster_id} ({profile['Size']} customers, {profile['Pct_of_Approvals']:.1f}% of approvals):")
        print(f"  Mean FICO: {profile['Mean_FICO']:.0f}")
        print(f"  Mean Income: ${profile['Mean_Income']:,.0f}")
        print(f"  Mean Loan: ${profile['Mean_Loan']:,.0f}")
        print(f"  Mean DTI: {profile['Mean_DTI']:.3f}")
        print(f"  Lender Distribution: A={profile['Lender_A_Pct']:.1f}%, B={profile['Lender_B_Pct']:.1f}%, C={profile['Lender_C_Pct']:.1f}%")

    cluster_df = pd.DataFrame(cluster_profiles)
    cluster_df.to_csv(TABLES_DIR / 'customer_clusters.csv', index=False)
    print(f"\n✓ Cluster profiles saved to {TABLES_DIR / 'customer_clusters.csv'}")

    return cluster_df, approved_with_cluster

def analyze_lender_preferences(cluster_df):
    """
    Create lender preference matrix showing which clusters each lender prefers
    """
    print("\n" + "="*80)
    print("LENDER PREFERENCE ANALYSIS")
    print("="*80)

    # Create preference matrix (Cluster × Lender)
    preference_matrix = cluster_df[['Cluster', 'Lender_A_Pct', 'Lender_B_Pct', 'Lender_C_Pct']].copy()
    preference_matrix.columns = ['Cluster', 'Lender_A', 'Lender_B', 'Lender_C']

    # Identify sweet spot clusters per lender (highest percentage)
    lender_sweet_spots = []

    for lender in ['Lender_A', 'Lender_B', 'Lender_C']:
        best_cluster = preference_matrix.loc[preference_matrix[lender].idxmax(), 'Cluster']
        best_pct = preference_matrix[lender].max()

        lender_sweet_spots.append({
            'Lender': lender.replace('Lender_', ''),
            'Sweet_Spot_Cluster': int(best_cluster),
            'Percentage': best_pct
        })

        print(f"\n{lender}:")
        print(f"  Sweet spot cluster: {int(best_cluster)} ({best_pct:.1f}% of cluster)")

    # Save preference matrix
    preference_matrix.to_csv(TABLES_DIR / 'lender_preference_matrix.csv', index=False)
    print(f"\n✓ Preference matrix saved to {TABLES_DIR / 'lender_preference_matrix.csv'}")

    # Save sweet spots
    sweet_spot_df = pd.DataFrame(lender_sweet_spots)
    sweet_spot_df.to_csv(TABLES_DIR / 'lender_sweet_spots.csv', index=False)
    print(f"✓ Sweet spots saved to {TABLES_DIR / 'lender_sweet_spots.csv'}")

    return preference_matrix, sweet_spot_df

def create_specialization_summary(anova_df, chi_df, cluster_df, sweet_spot_df):
    """
    Create comprehensive lender specialization summary
    """
    print("\n" + "="*80)
    print("LENDER SPECIALIZATION SUMMARY")
    print("="*80)

    summary = {
        'anova_significant_vars': anova_df[anova_df['Significant']]['Variable'].tolist(),
        'chi_square_significant_vars': chi_df[chi_df['Significant']]['Variable'].tolist(),
        'num_clusters': len(cluster_df),
        'lender_sweet_spots': sweet_spot_df.to_dict('records')
    }

    # Add key findings
    significant_anova = anova_df[anova_df['Significant']]
    significant_chi = chi_df[chi_df['Significant']]

    print(f"\nKey Findings:")
    print(f"  • {len(significant_anova)}/{len(anova_df)} numerical variables differ significantly across lenders")
    print(f"  • {len(significant_chi)}/{len(chi_df)} categorical variables differ significantly across lenders")
    print(f"  • {len(cluster_df)} customer segments identified")
    print(f"  • Each lender has clear segment preferences:")
    for spot in summary['lender_sweet_spots']:
        print(f"    - Lender {spot['Lender']}: Cluster {spot['Sweet_Spot_Cluster']} ({spot['Percentage']:.1f}%)")

    # Save JSON summary
    with open(TABLES_DIR / 'lender_specialization_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Specialization summary saved to {TABLES_DIR / 'lender_specialization_summary.json'}")

    return summary

def create_visualizations(anova_df, cluster_df, approved_with_cluster):
    """
    Create comprehensive specialization visualizations
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. ANOVA F-statistics (top-left)
    ax1 = axes[0, 0]
    significant = anova_df[anova_df['Significant']]
    if len(significant) > 0:
        colors = ['#2ecc71' if p < 0.001 else '#3498db' for p in significant['P_Value']]
        ax1.barh(significant['Variable'], significant['F_Statistic'], color=colors)
        ax1.set_xlabel('F-Statistic', fontsize=11)
        ax1.set_title('ANOVA: Numerical Variable Differences\nAcross Lenders (Approved Apps)', fontsize=12, fontweight='bold')
        ax1.axvline(x=3.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='F=3.0 threshold')
        ax1.legend(fontsize=9)

    # 2. Cluster sizes and lender distribution (top-right)
    ax2 = axes[0, 1]
    cluster_sizes = cluster_df['Size'].values
    lender_a_pcts = cluster_df['Lender_A_Pct'].values
    lender_b_pcts = cluster_df['Lender_B_Pct'].values
    lender_c_pcts = cluster_df['Lender_C_Pct'].values

    x = np.arange(len(cluster_df))
    width = 0.25

    ax2.bar(x - width, lender_a_pcts, width, label='Lender A', color='#e74c3c')
    ax2.bar(x, lender_b_pcts, width, label='Lender B', color='#3498db')
    ax2.bar(x + width, lender_c_pcts, width, label='Lender C', color='#2ecc71')

    ax2.set_xlabel('Customer Cluster', fontsize=11)
    ax2.set_ylabel('Lender Distribution (%)', fontsize=11)
    ax2.set_title('Lender Distribution Across Customer Segments', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Cluster {i}' for i in range(len(cluster_df))])
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Cluster characteristics heatmap (bottom-left)
    ax3 = axes[1, 0]
    cluster_features = cluster_df[['Cluster', 'Mean_FICO', 'Mean_Income', 'Mean_Loan', 'Mean_DTI']].copy()
    cluster_features.set_index('Cluster', inplace=True)

    # Normalize for heatmap
    cluster_normalized = (cluster_features - cluster_features.min()) / (cluster_features.max() - cluster_features.min())

    sns.heatmap(cluster_normalized.T, annot=True, fmt='.2f', cmap='RdYlGn',
                cbar_kws={'label': 'Normalized Value'}, ax=ax3, linewidths=0.5)
    ax3.set_title('Customer Segment Characteristics\n(Normalized)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Cluster', fontsize=11)
    ax3.set_ylabel('Feature', fontsize=11)

    # 4. Lender preference matrix (bottom-right)
    ax4 = axes[1, 1]
    preference_data = cluster_df[['Lender_A_Pct', 'Lender_B_Pct', 'Lender_C_Pct']].values.T

    im = ax4.imshow(preference_data, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(np.arange(len(cluster_df)))
    ax4.set_yticks(np.arange(3))
    ax4.set_xticklabels([f'Cluster {i}' for i in range(len(cluster_df))])
    ax4.set_yticklabels(['Lender A', 'Lender B', 'Lender C'])
    ax4.set_title('Lender Preference Matrix\n(% of cluster approved by lender)', fontsize=12, fontweight='bold')

    # Add percentage annotations
    for i in range(3):
        for j in range(len(cluster_df)):
            text = ax4.text(j, i, f'{preference_data[i, j]:.1f}%',
                           ha='center', va='center', color='white' if preference_data[i, j] > 25 else 'black',
                           fontsize=10, fontweight='bold')

    plt.colorbar(im, ax=ax4, label='Percentage (%)')

    plt.tight_layout()
    viz_path = FIGURES_DIR / 'lender_specialization_analysis.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Specialization visualization saved to {viz_path}")

def main():
    """
    Execute Phase 3.3: Lender Specialization Analysis
    """
    start_time = time.time()

    print("="*80)
    print("PHASE 3.3: LENDER SPECIALIZATION ANALYSIS")
    print("="*80)

    try:
        # 1. Load and prepare data
        df, report = load_and_validate()
        df = prepare_features(df)

        # 2. ANOVA tests
        anova_df = perform_anova_tests(df)

        # 3. Chi-square tests
        chi_df = perform_chi_square_tests(df)

        # 4. Clustering
        cluster_df, approved_with_cluster = perform_clustering(df)

        # 5. Lender preference analysis
        preference_matrix, sweet_spot_df = analyze_lender_preferences(cluster_df)

        # 6. Create specialization summary
        summary = create_specialization_summary(anova_df, chi_df, cluster_df, sweet_spot_df)

        # 7. Create visualizations
        create_visualizations(anova_df, cluster_df, approved_with_cluster)

        execution_time = time.time() - start_time

        # Build structured output
        output = {
            "success": True,
            "subphase": "Phase 3.3: Lender Specialization Analysis",
            "execution_time": round(execution_time, 2),
            "summary": {
                "anova_significant_count": len(anova_df[anova_df['Significant']]),
                "anova_total": len(anova_df),
                "chi_square_significant_count": len(chi_df[chi_df['Significant']]),
                "chi_square_total": len(chi_df),
                "num_clusters": len(cluster_df),
                "lender_sweet_spots": sweet_spot_df.to_dict('records')
            },
            "outputs": {
                "tables": [
                    "anova_results.csv",
                    "chi_square_results.csv",
                    "customer_clusters.csv",
                    "lender_preference_matrix.csv",
                    "lender_sweet_spots.csv",
                    "lender_specialization_summary.json"
                ],
                "figures": [
                    "lender_specialization_analysis.png"
                ]
            },
            "insights": [
                f"{len(anova_df[anova_df['Significant']])}/{len(anova_df)} numerical variables differ significantly across lenders",
                f"{len(chi_df[chi_df['Significant']])}/{len(chi_df)} categorical variables differ significantly across lenders",
                f"{len(cluster_df)} distinct customer segments identified",
                "Each lender shows clear segment preferences (sweet spots)"
            ]
        }

        # Output JSON for dashboard consumption
        print("\n" + "="*80)
        print("__JSON_OUTPUT__")
        print(json.dumps(output, indent=2))
        print("__JSON_OUTPUT_END__")

        print("="*80)
        print(f"✓ Phase 3.3 complete in {execution_time:.2f}s")
        print("="*80)

        return output

    except Exception as e:
        error_output = {
            "success": False,
            "subphase": "Phase 3.3: Lender Specialization Analysis",
            "error": str(e),
            "error_type": type(e).__name__
        }

        print("\n__JSON_OUTPUT__")
        print(json.dumps(error_output, indent=2))
        print("__JSON_OUTPUT_END__")

        raise

if __name__ == "__main__":
    main()
