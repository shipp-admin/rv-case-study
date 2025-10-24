"""
Phase 4.2: Optimal Matching Algorithm
Develops algorithm to assign customers to optimal lender for maximum expected revenue
Uses lender-specific models from Phase 3.2 to predict approval probabilities
"""

import sys
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple

warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from phase1_eda.data_loader import load_and_validate

# Setup paths
BASE_DIR = Path(__file__).parent.parent.parent
MODELS_DIR = BASE_DIR / 'models' / 'phase3_lender_models'
REPORTS_DIR = BASE_DIR / 'reports' / 'phase4_revenue_optimization'
TABLES_DIR = REPORTS_DIR / 'tables'
FIGURES_DIR = REPORTS_DIR / 'figures'

# Create directories
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Fixed bounty per approved application (from Phase 4.1 analysis)
BOUNTY_PER_APPROVAL = 240.66  # Mean bounty for approved apps

def load_lender_models() -> Dict[str, Dict]:
    """
    Load the 3 trained lender-specific models from Phase 3.2
    """
    print("\n" + "="*80)
    print("LOADING LENDER-SPECIFIC MODELS")
    print("="*80)

    models = {}
    lenders = ['a', 'b', 'c']
    lender_names = ['A', 'B', 'C']

    for lender, lender_name in zip(lenders, lender_names):
        model_path = MODELS_DIR / f'lender_{lender}_model.pkl'

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        models[lender_name] = {
            'model': model_data['model'],
            'feature_names': model_data['feature_names'],
            'metrics': model_data.get('metrics', {}),
            'best_params': model_data.get('best_params', {})
        }

        test_auc = model_data['metrics'].get('test_auc', 0)
        print(f"\n✓ Loaded Lender {lender_name} model:")
        print(f"  Test AUC: {test_auc:.4f}")
        print(f"  Features: {len(model_data['feature_names'])}")

    print(f"\n✓ All 3 lender models loaded successfully")
    return models

def prepare_features_for_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features exactly as they were for training in Phase 3.2
    """
    df = df.copy()

    # Engineered features
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

    # Custom FICO bins (matching Phase 3.2 training)
    df['FICO_Bin_Custom'] = pd.cut(
        df['FICO_score'],
        bins=[0, 600, 650, 700, 750, 850],
        labels=['very_low', 'low', 'medium', 'high', 'very_high']
    )

    # Encode categoricals (label encoding as used in training)
    df['Reason_encoded'] = df['Reason'].astype('category').cat.codes
    df['Employment_Status_encoded'] = df['Employment_Status'].astype('category').cat.codes
    df['Employment_Sector_encoded'] = df['Employment_Sector'].astype('category').cat.codes
    df['Ever_Bankrupt_or_Foreclose_encoded'] = df['Ever_Bankrupt_or_Foreclose'].astype('category').cat.codes
    df['Fico_Score_group_encoded'] = df['Fico_Score_group'].astype('category').cat.codes
    df['Income_Quartile_encoded'] = df['Income_Quartile'].astype('category').cat.codes
    df['FICO_Bin_Custom_encoded'] = df['FICO_Bin_Custom'].astype('category').cat.codes

    return df

def predict_approval_probabilities(df: pd.DataFrame, models: Dict[str, Dict]) -> pd.DataFrame:
    """
    Predict approval probability for each customer with each lender
    """
    print("\n" + "="*80)
    print("PREDICTING APPROVAL PROBABILITIES")
    print("="*80)

    predictions = pd.DataFrame(index=df.index)

    for lender_name in ['A', 'B', 'C']:
        model_data = models[lender_name]
        model = model_data['model']
        feature_names = model_data['feature_names']

        # Get features in correct order
        X = df[feature_names].copy()

        # Predict probabilities (probability of class 1 = approved)
        probas = model.predict_proba(X)[:, 1]
        predictions[f'P_approval_{lender_name}'] = probas

        print(f"\n✓ Predicted probabilities for Lender {lender_name}")
        print(f"  Mean P(approval): {probas.mean():.4f}")
        print(f"  Min: {probas.min():.4f}, Max: {probas.max():.4f}")

    return predictions

def calculate_expected_values(predictions: pd.DataFrame, bounty: float = BOUNTY_PER_APPROVAL) -> pd.DataFrame:
    """
    Calculate Expected Value = P(approval) × Bounty for each lender

    Mathematical Foundation:
    Expected Value (EV) is the average revenue we expect to earn from
    routing a customer to a specific lender.

    Formula: EV(customer, lender) = P(Approval | customer, lender) × Bounty

    Where:
    - P(Approval | customer, lender) = Probability of approval (0 to 1)
        Predicted by lender-specific Random Forest model
    - Bounty = Fixed revenue per approval = $240.66

    Example Calculation:
    - Customer with P(approval_A) = 0.18, P(approval_B) = 0.12, P(approval_C) = 0.21
    - Bounty = $240.66
    - EV_A = 0.18 × $240.66 = $43.32
    - EV_B = 0.12 × $240.66 = $28.88
    - EV_C = 0.21 × $240.66 = $50.54
    - Optimal assignment: Lender C (max EV = $50.54)

    Interpretation:
    - Higher EV = Higher expected revenue from that lender
    - EV accounts for both approval probability AND bounty amount
    - Used to find optimal lender assignment for each customer
    """
    print("\n" + "="*80)
    print("CALCULATING EXPECTED VALUES")
    print("="*80)

    ev_df = pd.DataFrame(index=predictions.index)

    # Calculate EV for each lender
    # EV = P(approval) × Bounty
    for lender in ['A', 'B', 'C']:
        ev_df[f'EV_{lender}'] = predictions[f'P_approval_{lender}'] * bounty

    print(f"\nExpected Values (using ${bounty:.2f} per approval):")
    for lender in ['A', 'B', 'C']:
        mean_ev = ev_df[f'EV_{lender}'].mean()
        print(f"  Lender {lender} mean EV: ${mean_ev:.2f}")

    return ev_df

def assign_optimal_lenders(ev_df: pd.DataFrame, predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Assign each customer to lender with maximum expected value

    Mathematical Foundation:
    For each customer, select the lender that maximizes expected revenue.
    This is a greedy optimization strategy.

    Formula: Optimal_Lender(customer) = argmax_j [EV(customer, lender_j)]

    Where:
    - argmax_j = "argument that maximizes" = lender with highest EV
    - j ∈ {A, B, C} = set of available lenders
    - EV(customer, lender_j) = expected value for customer with lender j

    Example:
    - Customer with EV_A = $43.32, EV_B = $28.88, EV_C = $50.54
    - max(43.32, 28.88, 50.54) = 50.54
    - argmax = Lender C
    - Optimal assignment: Customer → Lender C

    Assignment Confidence:
    - Confidence = (Best_EV - Second_Best_EV) / Best_EV
    - High confidence (>0.3): Clear winner, strong preference
    - Low confidence (<0.1): Multiple lenders equally good
    - Used to assess assignment stability

    Interpretation:
    - Each customer assigned to their optimal lender
    - No capacity constraints (greedy assignment)
    - Optimal_EV = maximum revenue we can expect from this customer
    - Sum of all Optimal_EV = total optimal revenue
    """
    print("\n" + "="*80)
    print("ASSIGNING OPTIMAL LENDERS")
    print("="*80)

    assignments = pd.DataFrame(index=ev_df.index)

    # Find lender with max EV for each customer
    # idxmax returns column name with maximum value per row
    ev_cols = [f'EV_{l}' for l in ['A', 'B', 'C']]
    assignments['Optimal_Lender'] = ev_df[ev_cols].idxmax(axis=1).str.replace('EV_', '')

    # Record the optimal EV and probability
    # max() returns the maximum value per row
    assignments['Optimal_EV'] = ev_df[ev_cols].max(axis=1)
    assignments['Optimal_P_Approval'] = predictions[[f'P_approval_{l}' for l in ['A', 'B', 'C']]].max(axis=1)

    # Calculate assignment confidence
    # Confidence = (Best_EV - Second_Best_EV) / Best_EV
    # Measures how much better the best lender is vs second-best
    ev_sorted = np.sort(ev_df[ev_cols].values, axis=1)[:, ::-1]  # Sort descending
    assignments['EV_Difference'] = ev_sorted[:, 0] - ev_sorted[:, 1]  # Best - 2nd best
    assignments['Assignment_Confidence'] = assignments['EV_Difference'] / assignments['Optimal_EV']

    # Record all EVs for comparison
    for lender in ['A', 'B', 'C']:
        assignments[f'EV_{lender}'] = ev_df[f'EV_{lender}']

    print(f"\n✓ Optimal assignments completed")
    print(f"\nDistribution of optimal assignments:")
    for lender in ['A', 'B', 'C']:
        count = (assignments['Optimal_Lender'] == lender).sum()
        pct = count / len(assignments) * 100
        print(f"  Lender {lender}: {count:,} ({pct:.1f}%)")

    print(f"\nAssignment Confidence Statistics:")
    print(f"  Mean confidence: {assignments['Assignment_Confidence'].mean():.4f}")
    print(f"  Median confidence: {assignments['Assignment_Confidence'].median():.4f}")
    print(f"  Low confidence (<0.1): {(assignments['Assignment_Confidence'] < 0.1).sum():,} applications")

    return assignments

def compare_optimal_vs_current(df: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
    """
    Compare optimal assignments to current (historical) assignments
    """
    print("\n" + "="*80)
    print("COMPARING OPTIMAL VS CURRENT ASSIGNMENTS")
    print("="*80)

    comparison = pd.DataFrame(index=df.index)
    comparison['Current_Lender'] = df['Lender']
    comparison['Optimal_Lender'] = assignments['Optimal_Lender']
    comparison['Should_Switch'] = comparison['Current_Lender'] != comparison['Optimal_Lender']

    # Calculate switch statistics
    total_apps = len(comparison)
    should_switch = comparison['Should_Switch'].sum()
    pct_switch = should_switch / total_apps * 100

    print(f"\nSwitch Analysis:")
    print(f"  Total applications: {total_apps:,}")
    print(f"  Should switch: {should_switch:,} ({pct_switch:.1f}%)")
    print(f"  Stay with current: {total_apps - should_switch:,} ({100-pct_switch:.1f}%)")

    # Switch matrix: From → To
    print(f"\nSwitch Matrix (Current → Optimal):")
    switch_matrix = pd.crosstab(
        comparison['Current_Lender'],
        comparison['Optimal_Lender'],
        margins=True
    )
    print(switch_matrix)

    # Save comparison
    comparison_summary = {
        'Total_Applications': int(total_apps),
        'Should_Switch': int(should_switch),
        'Pct_Switch': float(pct_switch),
        'Stay_Current': int(total_apps - should_switch)
    }

    # Add per-lender switching
    for lender in ['A', 'B', 'C']:
        current_lender = comparison[comparison['Current_Lender'] == lender]
        switch_count = current_lender['Should_Switch'].sum()
        comparison_summary[f'Switch_From_{lender}'] = int(switch_count)
        comparison_summary[f'Pct_Switch_From_{lender}'] = float(switch_count / len(current_lender) * 100) if len(current_lender) > 0 else 0.0

    comparison_df = pd.DataFrame([comparison_summary])
    comparison_df.to_csv(TABLES_DIR / 'optimal_vs_current_comparison.csv', index=False)
    print(f"\n✓ Comparison saved to {TABLES_DIR / 'optimal_vs_current_comparison.csv'}")

    return comparison

def analyze_performance(assignments: pd.DataFrame) -> Dict:
    """
    Analyze algorithm performance metrics
    """
    print("\n" + "="*80)
    print("ALGORITHM PERFORMANCE ANALYSIS")
    print("="*80)

    performance = {
        'total_assignments': len(assignments),
        'mean_optimal_ev': float(assignments['Optimal_EV'].mean()),
        'median_optimal_ev': float(assignments['Optimal_EV'].median()),
        'mean_confidence': float(assignments['Assignment_Confidence'].mean()),
        'median_confidence': float(assignments['Assignment_Confidence'].median()),
        'low_confidence_count': int((assignments['Assignment_Confidence'] < 0.1).sum()),
        'high_confidence_count': int((assignments['Assignment_Confidence'] >= 0.5).sum()),
        'lender_distribution': {
            'A': int((assignments['Optimal_Lender'] == 'A').sum()),
            'B': int((assignments['Optimal_Lender'] == 'B').sum()),
            'C': int((assignments['Optimal_Lender'] == 'C').sum())
        }
    }

    print(f"\nPerformance Metrics:")
    print(f"  Total Assignments: {performance['total_assignments']:,}")
    print(f"  Mean Optimal EV: ${performance['mean_optimal_ev']:.2f}")
    print(f"  Mean Assignment Confidence: {performance['mean_confidence']:.4f}")
    print(f"  Low Confidence (<0.1): {performance['low_confidence_count']:,}")
    print(f"  High Confidence (≥0.5): {performance['high_confidence_count']:,}")

    # Save performance metrics
    performance_df = pd.DataFrame([performance])
    performance_df.to_csv(TABLES_DIR / 'matching_algorithm_performance.csv', index=False)
    print(f"\n✓ Performance metrics saved to {TABLES_DIR / 'matching_algorithm_performance.csv'}")

    return performance

def create_visualizations(assignments: pd.DataFrame, comparison: pd.DataFrame, df: pd.DataFrame):
    """
    Create comprehensive matching algorithm visualizations
    """
    print("\n" + "="*80)
    print("CREATING MATCHING ALGORITHM VISUALIZATIONS")
    print("="*80)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. Current vs Optimal Distribution (top-left)
    ax1 = axes[0, 0]
    current_dist = df['Lender'].value_counts().sort_index()
    optimal_dist = assignments['Optimal_Lender'].value_counts().sort_index()

    x = np.arange(3)
    width = 0.35
    ax1.bar(x - width/2, current_dist.values, width, label='Current', color='#3498db', alpha=0.8)
    ax1.bar(x + width/2, optimal_dist.values, width, label='Optimal', color='#2ecc71', alpha=0.8)
    ax1.set_xlabel('Lender', fontsize=11)
    ax1.set_ylabel('Number of Applications', fontsize=11)
    ax1.set_title('Current vs Optimal Lender Distribution', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['A', 'B', 'C'])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2. Assignment Confidence Distribution (top-middle)
    ax2 = axes[0, 1]
    ax2.hist(assignments['Assignment_Confidence'], bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0.1, color='red', linestyle='--', label='Low Confidence (<0.1)')
    ax2.axvline(x=0.5, color='green', linestyle='--', label='High Confidence (≥0.5)')
    ax2.set_xlabel('Assignment Confidence', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Distribution of Assignment Confidence', fontsize=12, fontweight='bold')
    ax2.legend()

    # 3. Switch Analysis Pie Chart (top-right)
    ax3 = axes[0, 2]
    switch_counts = [
        comparison['Should_Switch'].sum(),
        (~comparison['Should_Switch']).sum()
    ]
    ax3.pie(switch_counts, labels=['Switch Lender', 'Stay Current'],
            autopct='%1.1f%%', colors=['#e74c3c', '#95a5a6'], startangle=90)
    ax3.set_title('Optimal Reassignment Analysis', fontsize=12, fontweight='bold')

    # 4. Expected Value by Lender (bottom-left)
    ax4 = axes[1, 0]
    ev_data = [assignments['EV_A'].values, assignments['EV_B'].values, assignments['EV_C'].values]
    bp = ax4.boxplot(ev_data, labels=['A', 'B', 'C'], patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#e74c3c', '#3498db', '#2ecc71']):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax4.set_xlabel('Lender', fontsize=11)
    ax4.set_ylabel('Expected Value ($)', fontsize=11)
    ax4.set_title('Expected Value Distribution by Lender', fontsize=12, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    # 5. Switch Matrix Heatmap (bottom-middle)
    ax5 = axes[1, 1]
    switch_matrix = pd.crosstab(comparison['Current_Lender'], comparison['Optimal_Lender'])
    sns.heatmap(switch_matrix, annot=True, fmt='d', cmap='YlOrRd', ax=ax5, cbar_kws={'label': 'Applications'})
    ax5.set_xlabel('Optimal Lender', fontsize=11)
    ax5.set_ylabel('Current Lender', fontsize=11)
    ax5.set_title('Lender Switch Matrix', fontsize=12, fontweight='bold')

    # 6. Optimal EV vs Confidence Scatter (bottom-right)
    ax6 = axes[1, 2]
    scatter = ax6.scatter(assignments['Assignment_Confidence'], assignments['Optimal_EV'],
                         c=assignments['Optimal_Lender'].map({'A': 0, 'B': 1, 'C': 2}),
                         cmap='Set1', alpha=0.3, s=10)
    ax6.set_xlabel('Assignment Confidence', fontsize=11)
    ax6.set_ylabel('Optimal Expected Value ($)', fontsize=11)
    ax6.set_title('EV vs Assignment Confidence', fontsize=12, fontweight='bold')
    ax6.grid(alpha=0.3)

    # Add colorbar for scatter
    cbar = plt.colorbar(scatter, ax=ax6, ticks=[0, 1, 2])
    cbar.set_label('Optimal Lender')
    cbar.set_ticklabels(['A', 'B', 'C'])

    plt.suptitle('Phase 4.2: Optimal Matching Algorithm Results', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    viz_path = FIGURES_DIR / 'matching_algorithm_results.png'
    plt.savefig(viz_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"✓ Matching algorithm visualization saved to {viz_path}")

def main():
    """
    Execute Phase 4.2: Optimal Matching Algorithm
    """
    start_time = time.time()

    print("="*80)
    print("PHASE 4.2: OPTIMAL MATCHING ALGORITHM")
    print("="*80)

    try:
        # 1. Load data
        df, report = load_and_validate()

        # 2. Load lender-specific models
        models = load_lender_models()

        # 3. Prepare features
        df = prepare_features_for_prediction(df)

        # 4. Predict approval probabilities for each lender
        predictions = predict_approval_probabilities(df, models)

        # 5. Calculate expected values
        ev_df = calculate_expected_values(predictions)

        # 6. Assign optimal lenders
        assignments = assign_optimal_lenders(ev_df, predictions)

        # 7. Compare optimal vs current
        comparison = compare_optimal_vs_current(df, assignments)

        # 8. Analyze performance
        performance = analyze_performance(assignments)

        # 9. Save assignments
        full_assignments = pd.concat([df[['Lender']], assignments], axis=1)
        full_assignments.to_csv(TABLES_DIR / 'optimal_lender_assignments.csv', index=False)
        print(f"\n✓ Full assignments saved to {TABLES_DIR / 'optimal_lender_assignments.csv'}")

        # 10. Create visualizations
        create_visualizations(assignments, comparison, df)

        execution_time = time.time() - start_time

        # Calculate per-customer latency
        latency_ms = (execution_time / len(df)) * 1000

        # Build structured output
        output = {
            "success": True,
            "subphase": "Phase 4.2: Optimal Matching Algorithm",
            "execution_time": round(execution_time, 2),
            "summary": {
                "total_applications": len(df),
                "mean_optimal_ev": round(float(assignments['Optimal_EV'].mean()), 2),
                "mean_assignment_confidence": round(float(assignments['Assignment_Confidence'].mean()), 4),
                "pct_should_switch": round(float(comparison['Should_Switch'].sum() / len(comparison) * 100), 2),
                "latency_per_customer_ms": round(latency_ms, 2),
                "optimal_distribution": {
                    "A": int((assignments['Optimal_Lender'] == 'A').sum()),
                    "B": int((assignments['Optimal_Lender'] == 'B').sum()),
                    "C": int((assignments['Optimal_Lender'] == 'C').sum())
                }
            },
            "outputs": {
                "tables": [
                    "optimal_lender_assignments.csv",
                    "optimal_vs_current_comparison.csv",
                    "matching_algorithm_performance.csv"
                ],
                "figures": [
                    "matching_algorithm_results.png"
                ]
            },
            "insights": [
                f"Optimal matching assigns {(assignments['Optimal_Lender'] == 'A').sum():,} to Lender A, {(assignments['Optimal_Lender'] == 'B').sum():,} to B, {(assignments['Optimal_Lender'] == 'C').sum():,} to C",
                f"{comparison['Should_Switch'].sum():,} applications ({comparison['Should_Switch'].sum()/len(comparison)*100:.1f}%) should switch lenders",
                f"Mean optimal expected value: ${assignments['Optimal_EV'].mean():.2f} per application",
                f"Assignment confidence: {assignments['Assignment_Confidence'].mean():.2%} average",
                f"Algorithm latency: {latency_ms:.2f}ms per customer (target: <50ms)"
            ]
        }

        # Output JSON for dashboard consumption
        print("\n" + "="*80)
        print("__JSON_OUTPUT__")
        print(json.dumps(output, indent=2))
        print("__JSON_OUTPUT_END__")

        print("="*80)
        print(f"✓ Phase 4.2 complete in {execution_time:.2f}s")
        print("="*80)

        return output

    except Exception as e:
        error_output = {
            "success": False,
            "subphase": "Phase 4.2: Optimal Matching Algorithm",
            "error": str(e),
            "error_type": type(e).__name__
        }

        print("\n__JSON_OUTPUT__")
        print(json.dumps(error_output, indent=2))
        print("__JSON_OUTPUT_END__")

        raise

if __name__ == "__main__":
    main()
