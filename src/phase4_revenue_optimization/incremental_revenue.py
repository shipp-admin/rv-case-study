"""
Phase 4.3: Incremental Revenue Calculation
Quantifies revenue lift from optimal matching vs. current state
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats

# Configuration
REPORTS_DIR = Path("reports/phase4_revenue_optimization")
TABLES_DIR = REPORTS_DIR / "tables"
FIGURES_DIR = REPORTS_DIR / "figures"
BOUNTY_PER_APPROVAL = 240.66  # From Phase 4.1

# Ensure directories exist
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_baseline_and_optimal_data():
    """
    Load baseline revenue and optimal assignments data
    """
    print("\n" + "="*80)
    print("LOADING BASELINE AND OPTIMAL DATA")
    print("="*80)

    # Load original data for customer features
    from src.phase1_eda.data_loader import load_and_validate
    original_df, _ = load_and_validate()

    # Load optimal assignments from Phase 4.2
    assignments_df = pd.read_csv(TABLES_DIR / "optimal_lender_assignments.csv")

    print(f"\n✓ Loaded optimal assignments: {len(assignments_df):,} rows")

    # Merge to get customer features
    # Assignments are in the same order as original data, so we can just add columns
    for col in ['Fico_Score_group', 'Monthly_Gross_Income', 'Loan_Amount']:
        if col in original_df.columns:
            assignments_df[col] = original_df[col].values

    print(f"  Merged with original features: {list(assignments_df.columns)}")

    # Load baseline RPA data from Phase 4.1
    baseline_overall = pd.read_csv(TABLES_DIR / "overall_rpa.csv")

    print(f"\n✓ Loaded baseline metrics")
    print(f"  Overall RPA: ${baseline_overall['Overall_RPA'].iloc[0]:.2f}")
    print(f"  Total Revenue: ${baseline_overall['Total_Bounty'].iloc[0]:,.0f}")

    return assignments_df, baseline_overall


def calculate_baseline_revenue(assignments_df, baseline_overall):
    """
    Calculate baseline (current state) revenue metrics
    """
    print("\n" + "="*80)
    print("CALCULATING BASELINE REVENUE")
    print("="*80)

    # Current lender assignments
    baseline_revenue = {
        'total_applications': len(assignments_df),
        'total_revenue': float(baseline_overall['Total_Bounty'].iloc[0]),
        'overall_rpa': float(baseline_overall['Overall_RPA'].iloc[0]),
        'approval_rate': float(baseline_overall['Approval_Rate'].iloc[0])
    }

    # Revenue by current lender
    lender_revenue = {}
    for lender in ['A', 'B', 'C']:
        lender_apps = assignments_df[assignments_df['Lender'] == lender]
        lender_revenue[lender] = {
            'applications': len(lender_apps),
            'revenue': len(lender_apps) * baseline_revenue['overall_rpa']  # Approximate
        }

    print(f"\nBaseline Revenue Summary:")
    print(f"  Total Applications: {baseline_revenue['total_applications']:,}")
    print(f"  Total Revenue: ${baseline_revenue['total_revenue']:,.0f}")
    print(f"  Overall RPA: ${baseline_revenue['overall_rpa']:.2f}")
    print(f"  Approval Rate: {baseline_revenue['approval_rate']:.2%}")

    print(f"\nRevenue by Current Lender:")
    for lender, metrics in lender_revenue.items():
        print(f"  Lender {lender}: {metrics['applications']:,} apps, ${metrics['revenue']:,.0f}")

    return baseline_revenue, lender_revenue


def calculate_optimal_revenue(assignments_df):
    """
    Calculate optimal revenue using Expected Value assignments
    """
    print("\n" + "="*80)
    print("CALCULATING OPTIMAL REVENUE")
    print("="*80)

    # Optimal expected revenue = sum of all optimal EVs
    optimal_total_ev = assignments_df['Optimal_EV'].sum()
    optimal_rpa = optimal_total_ev / len(assignments_df)

    # Optimal distribution
    optimal_distribution = assignments_df['Optimal_Lender'].value_counts()

    # Revenue by optimal lender
    optimal_lender_revenue = {}
    for lender in ['A', 'B', 'C']:
        lender_assignments = assignments_df[assignments_df['Optimal_Lender'] == lender]
        lender_ev = lender_assignments['Optimal_EV'].sum()
        optimal_lender_revenue[lender] = {
            'applications': len(lender_assignments),
            'expected_revenue': lender_ev,
            'rpa': lender_ev / len(lender_assignments) if len(lender_assignments) > 0 else 0
        }

    print(f"\nOptimal Revenue Summary:")
    print(f"  Expected Total Revenue: ${optimal_total_ev:,.0f}")
    print(f"  Expected RPA: ${optimal_rpa:.2f}")

    print(f"\nRevenue by Optimal Lender:")
    for lender, metrics in optimal_lender_revenue.items():
        print(f"  Lender {lender}: {metrics['applications']:,} apps, "
              f"${metrics['expected_revenue']:,.0f} revenue, "
              f"${metrics['rpa']:.2f} RPA")

    return {
        'total_revenue': optimal_total_ev,
        'overall_rpa': optimal_rpa,
        'lender_revenue': optimal_lender_revenue
    }


def calculate_incremental_revenue(baseline_revenue, optimal_revenue):
    """
    Calculate incremental revenue: optimal - baseline

    Mathematical Foundation:
    Incremental revenue measures the additional revenue generated by
    switching from random (baseline) to optimal lender assignment.

    Formulas:
    1. Incremental_Revenue = Optimal_Revenue - Baseline_Revenue
    2. Lift% = (Incremental_Revenue / Baseline_Revenue) × 100
    3. Incremental_RPA = Optimal_RPA - Baseline_RPA

    Where:
    - Baseline_Revenue = Σ[all customers] Current_Assignment_Revenue
    - Optimal_Revenue = Σ[all customers] max[EV(customer, lender)]
    - RPA = Revenue Per Application = Total_Revenue / Total_Applications

    Example Calculation:
    - Baseline: $2,641,500 (current random assignment)
    - Optimal: $5,582,149 (optimal matching)
    - Incremental: $5,582,149 - $2,641,500 = $2,940,649
    - Lift%: ($2,940,649 / $2,641,500) × 100 = 111.3%

    Interpretation:
    - Incremental_Revenue = Additional $ we can make from better matching
    - Lift% = Percentage improvement in revenue
    - >50% lift = Strong opportunity
    - >100% lift = Transformative opportunity (more than doubling revenue)
    """
    print("\n" + "="*80)
    print("CALCULATING INCREMENTAL REVENUE")
    print("="*80)

    # Calculate the key incremental metrics
    incremental = {
        # Total additional revenue from optimal matching
        'revenue_lift': optimal_revenue['total_revenue'] - baseline_revenue['total_revenue'],

        # Additional revenue per application
        'rpa_lift': optimal_revenue['overall_rpa'] - baseline_revenue['overall_rpa'],

        # Percentage improvement: (New - Old) / Old × 100
        'lift_percentage': ((optimal_revenue['total_revenue'] / baseline_revenue['total_revenue']) - 1) * 100
    }

    print(f"\n💰 Incremental Revenue Analysis:")
    print(f"  Baseline Revenue:    ${baseline_revenue['total_revenue']:,.0f}")
    print(f"  Optimal Revenue:     ${optimal_revenue['total_revenue']:,.0f}")
    print(f"  Incremental Lift:    ${incremental['revenue_lift']:,.0f}")
    print(f"  Lift Percentage:     +{incremental['lift_percentage']:.1f}%")
    print(f"\n  Baseline RPA:        ${baseline_revenue['overall_rpa']:.2f}")
    print(f"  Optimal RPA:         ${optimal_revenue['overall_rpa']:.2f}")
    print(f"  RPA Lift:            ${incremental['rpa_lift']:.2f} (+{(incremental['rpa_lift']/baseline_revenue['overall_rpa'])*100:.1f}%)")

    return incremental


def calculate_incremental_by_segment(assignments_df, baseline_rpa):
    """
    Calculate incremental revenue by customer segment
    """
    print("\n" + "="*80)
    print("CALCULATING INCREMENTAL REVENUE BY SEGMENT")
    print("="*80)

    segments = []

    # By FICO Score Group
    for fico_group in assignments_df['Fico_Score_group'].unique():
        segment_df = assignments_df[assignments_df['Fico_Score_group'] == fico_group]
        baseline_segment_revenue = len(segment_df) * baseline_rpa
        optimal_segment_revenue = segment_df['Optimal_EV'].sum()

        segments.append({
            'segment_type': 'FICO Group',
            'segment_value': fico_group,
            'applications': len(segment_df),
            'baseline_revenue': baseline_segment_revenue,
            'optimal_revenue': optimal_segment_revenue,
            'incremental_revenue': optimal_segment_revenue - baseline_segment_revenue,
            'lift_percentage': ((optimal_segment_revenue / baseline_segment_revenue) - 1) * 100 if baseline_segment_revenue > 0 else 0
        })

    # By Income Quartile
    income_quartiles = pd.qcut(assignments_df['Monthly_Gross_Income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    assignments_df['Income_Quartile'] = income_quartiles

    for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
        segment_df = assignments_df[assignments_df['Income_Quartile'] == quartile]
        baseline_segment_revenue = len(segment_df) * baseline_rpa
        optimal_segment_revenue = segment_df['Optimal_EV'].sum()

        segments.append({
            'segment_type': 'Income Quartile',
            'segment_value': quartile,
            'applications': len(segment_df),
            'baseline_revenue': baseline_segment_revenue,
            'optimal_revenue': optimal_segment_revenue,
            'incremental_revenue': optimal_segment_revenue - baseline_segment_revenue,
            'lift_percentage': ((optimal_segment_revenue / baseline_segment_revenue) - 1) * 100 if baseline_segment_revenue > 0 else 0
        })

    # By Loan Size Bracket
    loan_brackets = pd.cut(assignments_df['Loan_Amount'], bins=[0, 30000, 60000, float('inf')], labels=['Small', 'Medium', 'Large'])
    assignments_df['Loan_Bracket'] = loan_brackets

    for bracket in ['Small', 'Medium', 'Large']:
        segment_df = assignments_df[assignments_df['Loan_Bracket'] == bracket]
        baseline_segment_revenue = len(segment_df) * baseline_rpa
        optimal_segment_revenue = segment_df['Optimal_EV'].sum()

        segments.append({
            'segment_type': 'Loan Bracket',
            'segment_value': bracket,
            'applications': len(segment_df),
            'baseline_revenue': baseline_segment_revenue,
            'optimal_revenue': optimal_segment_revenue,
            'incremental_revenue': optimal_segment_revenue - baseline_segment_revenue,
            'lift_percentage': ((optimal_segment_revenue / baseline_segment_revenue) - 1) * 100 if baseline_segment_revenue > 0 else 0
        })

    segments_df = pd.DataFrame(segments)

    # Display top segments by lift
    print(f"\n✓ Calculated incremental revenue for {len(segments_df)} segments")
    print(f"\nTop 5 Segments by Incremental Revenue:")
    top_segments = segments_df.nlargest(5, 'incremental_revenue')
    for idx, row in top_segments.iterrows():
        print(f"  {row['segment_type']}: {row['segment_value']}")
        print(f"    Apps: {row['applications']:,}, Lift: ${row['incremental_revenue']:,.0f} (+{row['lift_percentage']:.1f}%)")

    return segments_df


def bootstrap_confidence_intervals(assignments_df, baseline_rpa, n_iterations=1000):
    """
    Calculate 95% confidence intervals using bootstrap resampling

    Mathematical Foundation:
    Bootstrap is a resampling technique to estimate the sampling distribution
    of a statistic (incremental revenue) without assumptions about the
    underlying data distribution.

    Algorithm:
    1. For each iteration b = 1 to B (typically 1000):
       a. Sample n customers WITH REPLACEMENT from original dataset
       b. Calculate incremental_revenue_b for this bootstrap sample
       c. Store result

    2. Sort all B incremental revenue values
    3. 95% CI lower bound = 2.5th percentile of sorted values
    4. 95% CI upper bound = 97.5th percentile of sorted values

    Mathematical Properties:
    - Percentile Method CI: [q₀.₀₂₅, q₀.₉₇₅]
    - q_α = α-th quantile of bootstrap distribution
    - WITH REPLACEMENT: Same customer can appear 0, 1, or multiple times
    - n = original sample size maintained in each bootstrap sample

    Example:
    - Original: 100,000 customers, incremental = $2,940,649
    - Bootstrap iteration 1: Resample 100,000 (with replacement), get $2,938,122
    - Bootstrap iteration 2: Resample 100,000 (with replacement), get $2,945,891
    - ... (repeat 1000 times)
    - Sort 1000 values
    - 2.5th percentile: $2,913,044 (lower bound)
    - 97.5th percentile: $2,968,476 (upper bound)
    - 95% CI: [$2,913,044, $2,968,476]

    Interpretation:
    - 95% confident true incremental revenue is in this interval
    - Narrow CI (width < 5% of estimate) = High precision
    - Accounts for sampling variability in our dataset
    """
    print("\n" + "="*80)
    print("BOOTSTRAP CONFIDENCE INTERVALS")
    print("="*80)
    print(f"Running {n_iterations:,} bootstrap iterations...")

    # Set random seed for reproducibility
    np.random.seed(42)
    incremental_revenues = []

    for i in range(n_iterations):
        # Step 1: Resample WITH REPLACEMENT
        # np.random.choice with replace=True allows same customer to appear multiple times
        sample_indices = np.random.choice(len(assignments_df), size=len(assignments_df), replace=True)
        sample_df = assignments_df.iloc[sample_indices]

        # Step 2: Calculate incremental revenue for this bootstrap sample
        baseline_sample = len(sample_df) * baseline_rpa
        optimal_sample = sample_df['Optimal_EV'].sum()
        incremental = optimal_sample - baseline_sample

        # Step 3: Store result
        incremental_revenues.append(incremental)

    # Calculate confidence intervals
    ci_lower = np.percentile(incremental_revenues, 2.5)
    ci_upper = np.percentile(incremental_revenues, 97.5)
    mean_incremental = np.mean(incremental_revenues)

    print(f"\n✓ Bootstrap complete")
    print(f"  Mean Incremental Revenue: ${mean_incremental:,.0f}")
    print(f"  95% Confidence Interval: [${ci_lower:,.0f}, ${ci_upper:,.0f}]")
    print(f"  CI Width: ${ci_upper - ci_lower:,.0f}")

    return {
        'mean': mean_incremental,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'iterations': n_iterations,
        'all_values': incremental_revenues
    }


def sensitivity_analysis(baseline_revenue, optimal_revenue, bounty_variations=[-0.10, -0.05, 0.0, 0.05, 0.10]):
    """
    Analyze sensitivity to bounty variations
    """
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS")
    print("="*80)
    print("Analyzing impact of bounty variations...")

    results = []
    baseline_total = baseline_revenue['total_revenue']

    for variation in bounty_variations:
        # Adjust bounty
        adjusted_bounty = BOUNTY_PER_APPROVAL * (1 + variation)

        # Optimal revenue scales with bounty
        adjusted_optimal = optimal_revenue['total_revenue'] * (1 + variation)

        # Baseline scales similarly
        adjusted_baseline = baseline_total * (1 + variation)

        incremental = adjusted_optimal - adjusted_baseline
        lift_pct = ((adjusted_optimal / adjusted_baseline) - 1) * 100

        results.append({
            'bounty_variation': f"{variation:+.0%}",
            'adjusted_bounty': adjusted_bounty,
            'baseline_revenue': adjusted_baseline,
            'optimal_revenue': adjusted_optimal,
            'incremental_revenue': incremental,
            'lift_percentage': lift_pct
        })

        print(f"  Bounty {variation:+.0%}: ${adjusted_bounty:.2f} → "
              f"Incremental: ${incremental:,.0f} (+{lift_pct:.1f}%)")

    return pd.DataFrame(results)


def create_visualizations(baseline_revenue, optimal_revenue, incremental, segments_df, bootstrap_results, sensitivity_df):
    """
    Create comprehensive visualization of incremental revenue analysis
    """
    print("\n" + "="*80)
    print("CREATING VISUALIZATIONS")
    print("="*80)

    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Baseline vs Optimal Revenue Comparison
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['Baseline', 'Optimal']
    revenues = [baseline_revenue['total_revenue'], optimal_revenue['total_revenue']]
    colors = ['#3498db', '#2ecc71']
    bars = ax1.bar(categories, revenues, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Total Revenue ($)', fontsize=11, fontweight='bold')
    ax1.set_title('Revenue Comparison', fontsize=12, fontweight='bold')
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:,.0f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 2. Incremental Revenue
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(['Incremental\nRevenue'], [incremental['revenue_lift']],
            color='#e74c3c', alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Revenue Lift ($)', fontsize=11, fontweight='bold')
    ax2.set_title(f"Revenue Lift: +{incremental['lift_percentage']:.1f}%",
                  fontsize=12, fontweight='bold')
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.2f}M'))
    ax2.text(0, incremental['revenue_lift'],
            f"${incremental['revenue_lift']:,.0f}",
            ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 3. RPA Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    rpas = [baseline_revenue['overall_rpa'], optimal_revenue['overall_rpa']]
    bars = ax3.bar(categories, rpas, color=colors, alpha=0.8, edgecolor='black')
    ax3.set_ylabel('Revenue Per Application ($)', fontsize=11, fontweight='bold')
    ax3.set_title('RPA Comparison', fontsize=12, fontweight='bold')
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'${height:.2f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 4. Top Segments by Lift
    ax4 = fig.add_subplot(gs[1, :2])
    top_10_segments = segments_df.nlargest(10, 'incremental_revenue')
    segment_labels = [f"{row['segment_type']}\n{row['segment_value']}"
                     for _, row in top_10_segments.iterrows()]
    ax4.barh(segment_labels, top_10_segments['incremental_revenue'],
            color='#9b59b6', alpha=0.8, edgecolor='black')
    ax4.set_xlabel('Incremental Revenue ($)', fontsize=11, fontweight='bold')
    ax4.set_title('Top 10 Segments by Incremental Revenue', fontsize=12, fontweight='bold')
    ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e3:.0f}K'))
    ax4.invert_yaxis()

    # 5. Bootstrap Confidence Interval
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.hist(bootstrap_results['all_values'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
    ax5.axvline(bootstrap_results['mean'], color='#e74c3c', linestyle='--', linewidth=2, label='Mean')
    ax5.axvline(bootstrap_results['ci_lower'], color='#f39c12', linestyle=':', linewidth=2, label='95% CI')
    ax5.axvline(bootstrap_results['ci_upper'], color='#f39c12', linestyle=':', linewidth=2)
    ax5.set_xlabel('Incremental Revenue ($)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax5.set_title('Bootstrap Distribution (1000 iterations)', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))

    # 6. Segment Lift Percentages
    ax6 = fig.add_subplot(gs[2, :2])
    segment_types = segments_df.groupby('segment_type')['lift_percentage'].mean().sort_values(ascending=True)
    ax6.barh(segment_types.index, segment_types.values, color='#1abc9c', alpha=0.8, edgecolor='black')
    ax6.set_xlabel('Average Lift %', fontsize=11, fontweight='bold')
    ax6.set_title('Average Lift by Segment Type', fontsize=12, fontweight='bold')
    ax6.axvline(incremental['lift_percentage'], color='#e74c3c', linestyle='--', linewidth=2, label='Overall')
    ax6.legend(fontsize=9)

    # 7. Sensitivity Analysis
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.plot(sensitivity_df['bounty_variation'], sensitivity_df['incremental_revenue'],
            marker='o', linewidth=2, markersize=8, color='#e74c3c')
    ax7.set_xlabel('Bounty Variation', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Incremental Revenue ($)', fontsize=11, fontweight='bold')
    ax7.set_title('Sensitivity to Bounty Changes', fontsize=12, fontweight='bold')
    ax7.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1e6:.1f}M'))
    ax7.grid(True, alpha=0.3)
    ax7.axhline(incremental['revenue_lift'], color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Phase 4.3: Incremental Revenue Analysis', fontsize=16, fontweight='bold', y=0.995)

    output_path = FIGURES_DIR / "incremental_revenue_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Visualization saved to {output_path}")


def save_results(baseline_revenue, optimal_revenue, incremental, segments_df, bootstrap_results, sensitivity_df):
    """
    Save all analysis results to CSV files
    """
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # Summary
    summary_df = pd.DataFrame([{
        'baseline_total_revenue': baseline_revenue['total_revenue'],
        'baseline_rpa': baseline_revenue['overall_rpa'],
        'optimal_total_revenue': optimal_revenue['total_revenue'],
        'optimal_rpa': optimal_revenue['overall_rpa'],
        'incremental_revenue': incremental['revenue_lift'],
        'incremental_rpa': incremental['rpa_lift'],
        'lift_percentage': incremental['lift_percentage'],
        'ci_lower': bootstrap_results['ci_lower'],
        'ci_upper': bootstrap_results['ci_upper'],
        'bootstrap_iterations': bootstrap_results['iterations']
    }])
    summary_path = TABLES_DIR / "incremental_revenue_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary saved to {summary_path}")

    # Segments
    segments_path = TABLES_DIR / "incremental_revenue_by_segment.csv"
    segments_df.to_csv(segments_path, index=False)
    print(f"✓ Segments saved to {segments_path}")

    # Bootstrap results
    bootstrap_df = pd.DataFrame({
        'iteration': range(len(bootstrap_results['all_values'])),
        'incremental_revenue': bootstrap_results['all_values']
    })
    bootstrap_path = TABLES_DIR / "bootstrap_confidence_intervals.csv"
    bootstrap_df.to_csv(bootstrap_path, index=False)
    print(f"✓ Bootstrap results saved to {bootstrap_path}")

    # Sensitivity
    sensitivity_path = TABLES_DIR / "sensitivity_analysis.csv"
    sensitivity_df.to_csv(sensitivity_path, index=False)
    print(f"✓ Sensitivity analysis saved to {sensitivity_path}")


def main():
    """
    Main execution function for Phase 4.3
    """
    print("="*80)
    print("PHASE 4.3: INCREMENTAL REVENUE CALCULATION")
    print("="*80)

    import time
    start_time = time.time()

    try:
        # 1. Load data
        assignments_df, baseline_overall = load_baseline_and_optimal_data()

        # 2. Calculate baseline revenue
        baseline_revenue, lender_revenue = calculate_baseline_revenue(assignments_df, baseline_overall)

        # 3. Calculate optimal revenue
        optimal_revenue = calculate_optimal_revenue(assignments_df)

        # 4. Calculate incremental revenue
        incremental = calculate_incremental_revenue(baseline_revenue, optimal_revenue)

        # 5. Segment analysis
        segments_df = calculate_incremental_by_segment(assignments_df, baseline_revenue['overall_rpa'])

        # 6. Bootstrap confidence intervals
        bootstrap_results = bootstrap_confidence_intervals(assignments_df, baseline_revenue['overall_rpa'])

        # 7. Sensitivity analysis
        sensitivity_df = sensitivity_analysis(baseline_revenue, optimal_revenue)

        # 8. Create visualizations
        create_visualizations(baseline_revenue, optimal_revenue, incremental,
                            segments_df, bootstrap_results, sensitivity_df)

        # 9. Save results
        save_results(baseline_revenue, optimal_revenue, incremental,
                    segments_df, bootstrap_results, sensitivity_df)

        execution_time = time.time() - start_time

        # Output JSON for dashboard
        json_output = {
            "success": True,
            "subphase": "Phase 4.3: Incremental Revenue Calculation",
            "execution_time": round(execution_time, 2),
            "summary": {
                "baseline_revenue": float(baseline_revenue['total_revenue']),
                "optimal_revenue": float(optimal_revenue['total_revenue']),
                "incremental_revenue": float(incremental['revenue_lift']),
                "lift_percentage": float(incremental['lift_percentage']),
                "baseline_rpa": float(baseline_revenue['overall_rpa']),
                "optimal_rpa": float(optimal_revenue['overall_rpa']),
                "rpa_lift": float(incremental['rpa_lift']),
                "ci_lower": float(bootstrap_results['ci_lower']),
                "ci_upper": float(bootstrap_results['ci_upper'])
            },
            "outputs": {
                "tables": [
                    "incremental_revenue_summary.csv",
                    "incremental_revenue_by_segment.csv",
                    "bootstrap_confidence_intervals.csv",
                    "sensitivity_analysis.csv"
                ],
                "figures": [
                    "incremental_revenue_analysis.png"
                ]
            },
            "insights": [
                f"Optimal matching increases revenue by ${incremental['revenue_lift']:,.0f} (+{incremental['lift_percentage']:.1f}%)",
                f"RPA increases from ${baseline_revenue['overall_rpa']:.2f} to ${optimal_revenue['overall_rpa']:.2f}",
                f"95% CI: [${bootstrap_results['ci_lower']:,.0f}, ${bootstrap_results['ci_upper']:,.0f}]",
                f"Top segment lift: {segments_df.nlargest(1, 'incremental_revenue').iloc[0]['segment_value']} "
                f"(+${segments_df.nlargest(1, 'incremental_revenue').iloc[0]['incremental_revenue']:,.0f})"
            ]
        }

        print("\n" + "="*80)
        print("__JSON_OUTPUT__")
        print(json.dumps(json_output, indent=2))
        print("__JSON_OUTPUT_END__")
        print("="*80)
        print(f"✓ Phase 4.3 complete in {execution_time:.2f}s")
        print("="*80)

    except Exception as e:
        print(f"\n❌ Error in Phase 4.3: {str(e)}")
        import traceback
        traceback.print_exc()

        json_output = {
            "success": False,
            "subphase": "Phase 4.3: Incremental Revenue Calculation",
            "error": str(e),
            "error_type": type(e).__name__
        }

        print("\n__JSON_OUTPUT__")
        print(json.dumps(json_output, indent=2))
        print("__JSON_OUTPUT_END__")


if __name__ == "__main__":
    main()
