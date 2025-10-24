# Product Requirements Document: Lending Approval & Revenue Optimization Analysis

**Document Version**: 1.0
**Date**: 2025-10-22
**Status**: Analysis Framework - Ready for Implementation

---

## Executive Summary

Analyze lending platform data to optimize customer-lender matching and maximize Revenue Per Application (RPA). The analysis will identify key approval drivers, characterize lender preferences, and develop an intelligent matching algorithm to increase incremental revenue through better customer-lender alignment.

**Key Value Proposition**: Data-driven lender matching that maximizes approval rates and revenue while maintaining risk standards across three lending partners (A, B, C).

**Dataset Overview**:
- **Total Applications**: 100,000
- **Time Period**: Historical dataset (candidate assessment)
- **Lenders**: 3 partners (A: 55%, B: 27.5%, C: 17.5% of volume)
- **Overall Approval Rate**: 10.98%
- **Revenue Model**: Bounty-based (paid on approvals only)

---

## Problem Statement

The lending platform currently routes customers to lenders without optimizing for:
1. **Approval Likelihood**: Customers may be sent to lenders likely to reject them
2. **Revenue Maximization**: Higher-bounty lenders may be underutilized for qualified customers
3. **Lender Preferences**: Each lender has unique approval criteria not being leveraged
4. **Feature Efficiency**: Collecting unnecessary data or missing predictive variables

**Business Impact**:
- Lost revenue from suboptimal lender matching
- Poor customer experience from avoidable rejections
- Operational inefficiency from collecting non-predictive variables
- Missed opportunities to scale high-performing lender relationships

**Target Stakeholders**: Product team, data science team, lending operations, partner relationship managers

---

## Research Questions

### Question 1: Variable Relationship with Approvability
**Objective**: Identify which variables are most predictive of loan approval and optimize data collection

**Sub-Questions**:
- Which variables have the strongest correlation with approval outcomes?
- Are there variables with no predictive power that can be eliminated?
- What feature transformations could improve predictive accuracy?
- Are there interaction effects between variables (e.g., FICO × Income)?
- Do different variables matter for different lenders?

**Success Metrics**:
- Feature importance rankings with statistical significance
- Predictive model accuracy (target: >75% AUC-ROC)
- Identified variables for removal (low importance)
- Engineered features with measurable lift

### Question 2: Lender Approval Rate Analysis
**Objective**: Profile each lender's approval patterns and preferences

**Sub-Questions**:
- What is each lender's baseline approval rate?
- Do lenders specialize in specific customer segments (FICO, income, employment)?
- Are there clear thresholds (e.g., minimum FICO score) per lender?
- Which variables reliably predict approval at each lender?
- Are there loan purposes or sectors certain lenders prefer/avoid?

**Success Metrics**:
- Approval rate by lender with confidence intervals
- Statistical tests showing significant differences between lenders
- Lender-specific predictive models (>70% accuracy)
- Identified "sweet spot" customer profiles per lender

### Question 3: Revenue Optimization via Intelligent Matching
**Objective**: Develop optimal customer-lender matching strategy to maximize Revenue Per Application

**Sub-Questions**:
- What is current Revenue Per Application by lender and segment?
- Which customer segments are mismatched to lenders?
- What is the incremental revenue opportunity from optimal matching?
- What are the real-time matching considerations (latency, fallback logic)?
- Should we prioritize approval rate or bounty value?

**Success Metrics**:
- Baseline RPA vs. optimized RPA comparison
- Incremental revenue calculation with confidence intervals
- Customer segments with highest re-routing potential
- Real-time matching algorithm design specifications
- A/B test design for production validation

---

## Data Overview

### Dataset Specifications
**File**: `Pre-Super_Day_candidate_dataset__28candidate_29 (1).xlsx`
**Location**: `./app/`
**Format**: Excel (.xlsx)
**Rows**: 100,000 loan applications
**Columns**: 14 variables

### Data Dictionary

| Variable | Type | Description | Example Values | Missing Values |
|----------|------|-------------|----------------|----------------|
| `User ID` | string | Unique customer identifier | UUID format | 0 |
| `applications` | int | Application count (always 1) | 1 | 0 |
| `Reason` | string | Loan purpose | debt_conslidation, home_improvement, major_purchase, credit_card_refinancing, cover_an_unexpected_cost, other | 0 |
| `Loan_Amount` | int | Requested loan value ($) | 10,000 - 100,000 | 0 |
| `FICO_score` | int | Credit risk score | 300-850 | 0 |
| `Fico_Score_group` | string | FICO category | poor, fair, good, very_good, exceptional | 0 |
| `Employment_Status` | string | Employment type | full_time, part_time | 0 |
| `Employment_Sector` | string | Industry classification | energy, materials, financials, consumer_discretionary, consumer_staples, information_technology, communication_services, utilities, health_care, real_estate | 6,407 (6.4%) |
| `Monthly_Gross_Income` | int | Pre-tax monthly income ($) | Range varies | 0 |
| `Monthly_Housing_Payment` | int | Monthly housing cost ($) | Range varies | 0 |
| `Ever_Bankrupt_or_Foreclose` | int | Binary flag (0/1) | 0 = No, 1 = Yes | 0 |
| `Lender` | string | Lending partner | A, B, C | 0 |
| `Approved` | int | Approval outcome | 0 = Denied, 1 = Approved | 0 |
| `bounty` | int | Revenue received ($) | $0 (denied), $250 (approved typical) | 0 |

### Initial Data Observations

**Approval Rates by Lender**:
| Lender | Applications | Approval Rate | Volume Share |
|--------|--------------|---------------|--------------|
| A | 55,000 | 10.97% | 55.0% |
| B | 27,500 | 7.13% | 27.5% |
| C | 17,500 | 17.06% | 17.5% |

**Key Insights**:
- Lender C has 2.4× higher approval rate than Lender B despite lower volume
- Lender A processes majority of applications with middle-tier approval rate
- Overall approval rate of 10.98% suggests selective lending criteria
- Employment Sector has 6.4% missing values (requires handling strategy)

**Data Quality Notes**:
- No missing values except Employment_Sector
- `applications` column always 1 (can be removed from analysis)
- FICO scores appear well-distributed across credit tiers
- Bounty values likely vary by lender (requires verification)

---

## Question-to-Phase Mapping

This table shows exactly which phases answer each original research question:

| Original Question | Analysis Phases | Key Deliverables |
|-------------------|-----------------|------------------|
| **Q1: Variable Relationship with Approvability** | | |
| → Which variables are most helpful? | Phase 1.1, 2.1, 2.2 | Feature importance rankings, top 5 predictive variables |
| → Which variables are not useful to collect? | Phase 2.1, 2.2 | List of features to drop (p>0.05, low importance) |
| → Feature transformations that improve predictive power? | Phase 1.2, 2.3 | Engineered features with AUC lift metrics |
| **Q2: Lender Approval Rates** | | |
| → What is each lender's average approval rate? | Phase 3.1 | Approval rates: A (10.97%), B (7.13%), C (17.06%) |
| → Clear differences between lenders on customer types? | Phase 3.1, 3.3 | Lender profile reports, ANOVA/chi-square results |
| → Variables that predict approval for particular lender? | Phase 3.2 | 3 lender-specific models + feature importance per lender |
| **Q3: Revenue Optimization via Matching** | | |
| → Current Revenue Per Application? | Phase 4.1 | Baseline RPA calculations by lender and segment |
| → Calculate incremental revenue from optimal matching | Phase 4.3 | Revenue lift estimate with 95% confidence interval |
| → Customer groups better fit for different lender? | Phase 4.2, 4.3 | Optimal matching algorithm + segment re-routing analysis |
| → Real-time matching considerations? | Phase 4.4, 4.5 | System architecture, API specs, A/B test design |

---

## Analysis Approach

> **📋 Question Tracking**: Each phase explicitly maps to the original research questions for easy progress tracking.

### Phase 1: Exploratory Data Analysis (EDA)
**Duration**: Week 1
**Owner**: Data Analyst
**Addresses**: Question 1 (preliminary analysis)

**Validation Checkpoints**:
- ✅ Data loads successfully with correct shape (100,000 rows × 13 columns after dropping `applications`)
- ✅ No data type errors or conversion issues
- ✅ Missing values documented (Employment_Sector: 6.4%)
- ✅ Approval rate matches expected baseline (10.98%)
- ✅ All visualizations render without errors
- ✅ Summary statistics match data dictionary ranges

**UI Development Pattern**:
Phase 1 follows a systematic pattern: **Implement → Validate → Display Results**

Each subphase (1.1, 1.2, 1.3) has:
1. **Implementation Script**: `src/phase1_eda/{subphase}.py`
2. **Validation Test**: `tests/test_phase1_{subphase}.py`
3. **Dashboard Integration**: Added to `ValidationSidebar` for run/validate
4. **Results Display**: Shared `ValidationResults` component shows figures/tables/insights

**Dashboard Features**:
- ✅ Sidebar buttons to Run/Validate each subphase
- ✅ Subphase tabs (1.1, 1.2, 1.3) to switch between results
- ✅ Console Output tab shows execution logs
- ✅ Results tab shows:
  - 📈 Figures gallery (interactive, click to enlarge)
  - 📊 Tables preview (click to view/download CSV)
  - 💡 Key insights from analysis
  - ⏱️ Execution time

**No separate Question 1 UI needed** - dashboard already displays all Phase 1 results per subphase.

#### 1.1 Univariate Analysis
**Objective**: Understand individual variable distributions and relationships with approval
**Answers**: Q1 - "Which variables correlate with approval?" (preliminary findings)

**Validation Criteria**:
- Approval rates calculated for all categorical variables
- Distributions plotted for all numerical variables
- No variables with >95% same value (except `applications`)
- Outliers identified and documented

**Tasks**:
- Calculate approval rates by each categorical variable (Reason, Employment_Status, Fico_Score_group, Lender)
- Plot distributions of numerical variables (Loan_Amount, FICO_score, Monthly_Gross_Income, Monthly_Housing_Payment)
- Create approval rate heatmaps for categorical variables
- Identify outliers and data quality issues

**Deliverables**:
- Distribution plots for all variables
- Approval rate tables by category
- Data quality report
- Initial hypotheses about predictive variables

**Tools**: Python (pandas, matplotlib, seaborn)

**Validation Test** (Run after completion):
```bash
# Verify subphase 1.1 outputs exist and are correct
python3 tests/test_phase1_univariate.py

# Expected outputs to verify:
# - reports/phase1_eda/figures/approval_by_fico.png exists
# - reports/phase1_eda/figures/income_distribution.png exists
# - reports/phase1_eda/tables/approval_rates_by_category.csv exists
# - All categorical variables have approval rates calculated
# - All numerical variables have distribution plots
```

#### 1.2 Bivariate Analysis
**Objective**: Identify correlations and relationships between variables
**Answers**: Q1 - "Are there feature transformations that would improve predictive power?" (feature engineering)

**Tasks**:
- Correlation matrix for numerical variables
- Chi-square tests for categorical variables vs. approval
- T-tests/ANOVA for numerical variables vs. approval
- Create derived features:
  - **Debt-to-Income Ratio**: `Monthly_Housing_Payment / Monthly_Gross_Income`
  - **Loan-to-Income Ratio**: `Loan_Amount / (Monthly_Gross_Income × 12)`
  - **FICO Bins**: Custom binning beyond standard groups
  - **Income Quartiles**: Segment income into Q1-Q4
  - **Loan Amount Categories**: Small (<$30K), Medium ($30-60K), Large (>$60K)

**Deliverables**:
- Correlation heatmap
- Statistical test results with p-values
- Engineered feature definitions
- Bivariate approval rate charts

**Tools**: Python (scipy, statsmodels)

**Validation Test** (Run after completion):
```bash
# Verify subphase 1.2 outputs exist and are correct
python3 tests/test_phase1_bivariate.py

# Expected outputs to verify:
# - reports/phase1_eda/figures/correlation_heatmap.png exists
# - reports/phase1_eda/tables/statistical_tests.csv exists (chi-square, t-test results)
# - data/processed/features_engineered.csv exists with DTI, LTI columns
# - All p-values calculated and significant relationships identified
# - Bivariate charts show variable interactions with approval
```

#### 1.3 Missing Value Treatment
**Objective**: Handle 6.4% missing Employment_Sector values
**Answers**: Data quality assurance for reliable analysis

**Strategy Options**:
1. **Imputation**: Use mode (most common sector) or predictive model
2. **Separate Category**: Create "Unknown" category
3. **Deletion**: Remove rows (if missing pattern not systematic)
4. **Model-Based**: Train classifier to predict sector from other variables

**Recommended**: Create "Unknown" category + test if missing pattern predicts approval

**Deliverables**:
- Missing value analysis report
- Chosen imputation strategy with justification
- Code for missing value handling

**Validation Test** (Run after completion):
```bash
# Verify subphase 1.3 outputs exist and are correct
python3 tests/test_phase1_missing_values.py

# Expected outputs to verify:
# - data/processed/cleaned_data.csv exists with NO missing Employment_Sector values
# - Missing value analysis report shows strategy (Unknown category OR imputation)
# - Test if missing pattern predicts approval (chi-square test result documented)
# - No other unexpected missing values introduced
# - Data shape remains 100,000 rows (no deletions unless justified)
```

### Phase 2: Feature Importance & Selection
**Duration**: Week 1-2
**Owner**: Data Scientist
**Addresses**: Question 1 (definitive variable importance rankings)

**UI Development Pattern** (Same as Phase 1):
Phase 2 follows the same systematic pattern: **Implement → Validate → Display Results**

Each subphase (2.1, 2.2, 2.3) will have:
1. **Implementation Script**: `src/phase2_feature_importance/{subphase}.py`
2. **Validation Test**: `tests/test_phase2_{subphase}.py`
3. **Dashboard Integration**: Added to `ValidationSidebar` under "Phase 2" section
4. **Results Display**: Same `ValidationResults` component shows all outputs

**Dashboard Updates Needed**:
- Add Phase 2 section to sidebar with subphases 2.1, 2.2, 2.3
- Add Phase 2 subphase tabs to main dashboard
- Results automatically display figures/tables from `reports/phase2_feature_importance/`

#### 2.1 Statistical Feature Importance
**Objective**: Rank variables by predictive power using statistical methods
**Answers**: Q1 - "Which variables are most helpful?" + "Which variables are not useful to collect?"

**Methods**:
1. **Mutual Information**: Measure mutual dependence between features and approval
2. **ANOVA F-statistic**: Test mean differences for numerical variables
3. **Chi-Square Test**: Independence testing for categorical variables
4. **Point-Biserial Correlation**: Correlation between binary approval and numerical features

**Deliverables**:
- Feature importance rankings
- Statistical significance table (p-values)
- Recommended features to drop (p > 0.05)

**Validation Test** (Run after completion):
```bash
# Verify subphase 2.1 outputs exist and are correct
python3 tests/test_phase2_statistical_importance.py

# Expected outputs to verify:
# - reports/phase2_feature_importance/tables/feature_rankings.csv exists
# - All features ranked by mutual information, ANOVA F-stat, chi-square
# - p-values < 0.05 for top 5 features
# - List of features to drop (p > 0.05) documented
# - Statistical test results match expected distributions
```

#### 2.2 Machine Learning Feature Importance
**Objective**: Use ML models to assess feature importance
**Answers**: Q1 - "Which variables are most helpful?" (ML-based validation of statistical findings)

**Models**:
1. **Random Forest Classifier**
   - Gini importance and permutation importance
   - Out-of-bag (OOB) error for validation
   - Feature importance plots

2. **XGBoost Classifier**
   - Gain, cover, and frequency importance metrics
   - SHAP values for feature interpretability
   - Early stopping with cross-validation

3. **Logistic Regression (L1 Regularization)**
   - Coefficient magnitudes
   - Automated feature selection via Lasso
   - Interpretable odds ratios

**Deliverables**:
- Feature importance plots from all models
- Consensus feature rankings
- SHAP summary plots
- Recommendations for feature engineering

**Tools**: Python (scikit-learn, xgboost, shap)

**Validation Test** (Run after completion):
```bash
# Verify subphase 2.2 outputs exist and are correct
python3 tests/test_phase2_ml_importance.py

# Expected outputs to verify:
# - reports/phase2_feature_importance/figures/feature_importance_rf.png exists
# - reports/phase2_feature_importance/figures/shap_summary.png exists
# - models/phase2_feature_models/rf_feature_selector.pkl exists
# - Consensus top 5 features identified across all models
# - AUC-ROC > 0.70 for baseline model
# - Feature importance scores normalized and comparable
```

#### 2.3 Feature Engineering Validation
**Objective**: Test if engineered features improve model performance
**Answers**: Q1 - "Are there feature transformations that improve predictive power?" (validation with metrics)

**Process**:
1. Train baseline model with original features
2. Train model with engineered features added
3. Compare AUC-ROC, precision, recall, F1-score
4. Validate on hold-out test set

**Deliverables**:
- Model performance comparison table
- Feature engineering recommendations
- Final feature set for production models

**Validation Test** (Run after completion):
```bash
# Verify subphase 2.3 outputs exist and are correct
python3 tests/test_phase2_feature_validation.py

# Expected outputs to verify:
# - reports/phase2_feature_importance/tables/model_comparison.csv exists
# - Baseline model AUC-ROC documented
# - Engineered features model AUC-ROC >= baseline + 0.03
# - Improvement metrics: precision, recall, F1 documented
# - Final feature set CSV with all recommended features
# - Hold-out test set performance validated
```

### Phase 3: Lender-Specific Analysis
**Duration**: Week 2-3
**Owner**: Data Analyst + Data Scientist
**Addresses**: Question 2 (complete lender analysis)

#### 3.1 Lender Approval Profiling
**Objective**: Characterize each lender's approval patterns
**Answers**: Q2 - "What is each lender's average approval rate?" + "Are there clear differences between lenders on customer types?"

**Analysis per Lender (A, B, C)**:
1. **Approval Rate by Segment**:
   - FICO score groups (poor, fair, good, very_good, exceptional)
   - Income quartiles
   - Loan amount brackets
   - Employment status (full_time, part_time)
   - Loan reason categories
   - Bankruptcy history

2. **Statistical Comparisons**:
   - Mean FICO score: Approved vs. Denied
   - Mean income: Approved vs. Denied
   - Mean loan amount: Approved vs. Denied
   - Mean debt-to-income ratio: Approved vs. Denied

3. **Threshold Detection**:
   - Minimum FICO score for approvals (5th percentile)
   - Minimum income for approvals
   - Maximum loan-to-income ratio for approvals
   - Bankruptcy tolerance (approval rate with bankruptcy flag)

**Deliverables**:
- Lender profile reports (3 reports: A, B, C)
- Segment approval rate heatmaps per lender
- Statistical comparison tables
- Identified approval thresholds per lender

**Tools**: Python (pandas, scipy)

**Validation Test** (Run after completion):
```bash
# Verify subphase 3.1 outputs exist and are correct
python3 tests/test_phase3_lender_profiling.py

# Expected outputs to verify:
# - reports/phase3_lender_analysis/tables/lender_approval_rates.csv exists
# - 3 lender profile reports (A, B, C) with segment breakdowns
# - Approval rate heatmaps for each lender exist
# - Approval thresholds identified (min FICO, max DTI, etc.)
# - Statistical tests confirm lenders differ significantly (p < 0.01)
```

#### 3.2 Lender-Specific Predictive Models
**Objective**: Build separate models to predict approval for each lender
**Answers**: Q2 - "Are there variables that reliably predict approval likelihood for a particular lender?"

**Approach**:
- Split data by lender (3 datasets)
- Train XGBoost classifier per lender
- Perform hyperparameter tuning (grid search)
- Cross-validate on 5 folds
- Calibrate probabilities (Platt scaling)

**Model Outputs**:
- **Lender A Model**: `P(approval | customer, lender=A)`
- **Lender B Model**: `P(approval | customer, lender=B)`
- **Lender C Model**: `P(approval | customer, lender=C)`

**Deliverables**:
- 3 trained models with calibrated probabilities
- Model performance metrics (AUC-ROC, precision-recall curves)
- Feature importance per lender
- Comparison of which features matter most for each lender

**Tools**: Python (xgboost, scikit-learn)

**Validation Test** (Run after completion):
```bash
# Verify subphase 3.2 outputs exist and are correct
python3 tests/test_phase3_lender_models.py

# Expected outputs to verify:
# - models/phase3_lender_models/lender_a_model.pkl exists and loads
# - models/phase3_lender_models/lender_b_model.pkl exists and loads
# - models/phase3_lender_models/lender_c_model.pkl exists and loads
# - All 3 models achieve AUC-ROC > 0.70 on test set
# - Calibration plots show well-calibrated probabilities
# - Feature importance differs meaningfully between lenders
# - Model metadata JSON documents all hyperparameters
```

#### 3.3 Lender Specialization Analysis
**Objective**: Identify clear differences in customer types each lender approves
**Answers**: Q2 - "Are there clear differences between lenders on what type of customers they approve?"

**Statistical Tests**:
1. **ANOVA**: Test if mean FICO, income, loan amounts differ across lenders for approvals
2. **Chi-Square**: Test if approval rates differ by sector, reason, employment status
3. **Post-hoc Tests**: Tukey HSD to identify pairwise differences

**Segmentation Analysis**:
- Cluster approved customers (k-means on FICO, income, loan amount)
- Analyze lender distribution within clusters
- Identify "sweet spot" clusters per lender

**Deliverables**:
- Statistical test results (ANOVA, chi-square tables)
- Lender specialization summary
- Customer segment definitions
- Lender preference matrix (segment × lender)

**Validation Test** (Run after completion):
```bash
# Verify subphase 3.3 outputs exist and are correct
python3 tests/test_phase3_specialization.py

# Expected outputs to verify:
# - reports/phase3_lender_analysis/tables/lender_specialization.csv exists
# - ANOVA results show significant differences (p < 0.05) in FICO, income, loan amounts
# - Chi-square tests show lender preferences by sector, reason, employment
# - Customer clusters identified (3-5 segments)
# - Lender preference matrix shows clear specialization patterns
# - "Sweet spot" segments identified for each lender
```

**UI Development Pattern** (Same as Phases 1 & 2):
Phase 3 follows the same systematic pattern: **Implement → Validate → Display Results**

Each subphase (3.1, 3.2, 3.3) will have:
1. **Implementation Script**: `src/phase3_lender_analysis/{subphase}.py`
2. **Validation Test**: `tests/test_phase3_{subphase}.py`
3. **Dashboard Integration**: Added to `ValidationSidebar` under "Phase 3" section
4. **Results Display**: Same `ValidationResults` component shows all outputs

**Dashboard Updates Needed**:
- Add Phase 3 section to sidebar with subphases 3.1, 3.2, 3.3
- Add Phase 3 subphase tabs to main dashboard
- Results automatically display figures/tables from `reports/phase3_lender_analysis/`

### Phase 4: Revenue Optimization Modeling
**Duration**: Week 3-4
**Owner**: Data Scientist + Product Analyst
**Addresses**: Question 3 (revenue maximization through optimal matching)

#### 4.1 Current State Revenue Analysis
**Objective**: Calculate baseline Revenue Per Application (RPA)
**Answers**: Q3 - "What is current Revenue Per Application by lender and segment?" (baseline establishment)

**Metrics**:
1. **Overall RPA**: `Total Bounty / Total Applications`
2. **RPA by Lender**: `Bounty per Lender / Applications per Lender`
3. **RPA by Customer Segment**: Segment by FICO, income, loan amount
4. **Bounty Distribution**: Analyze if bounty varies by lender or customer type

**Analysis**:
```python
# Verify bounty structure
bounty_by_lender = df.groupby('Lender')['bounty'].agg(['mean', 'sum', 'count'])
bounty_by_approval = df.groupby(['Lender', 'Approved'])['bounty'].mean()

# Current RPA
overall_rpa = df['bounty'].sum() / len(df)
rpa_by_lender = df.groupby('Lender')['bounty'].sum() / df.groupby('Lender').size()
```

**Deliverables**:
- Current state revenue dashboard
- RPA by lender and segment
- Bounty structure documentation
- Baseline revenue metrics

**Validation Test** (Run after completion):
```bash
# Verify subphase 4.1 outputs exist and are correct
python3 tests/test_phase4_baseline_revenue.py

# Expected outputs to verify:
# - reports/phase4_revenue_optimization/tables/baseline_rpa.csv exists
# - Overall RPA calculated and documented
# - RPA by lender (A, B, C) calculated
# - RPA by customer segment calculated
# - Bounty structure verified (fixed vs variable)
# - Revenue calculations match: Total Bounty / Total Applications
```

#### 4.2 Optimal Matching Algorithm
**Objective**: Develop algorithm to assign customers to lenders for maximum expected revenue
**Answers**: Q3 - "Which customers should we match to each lender?" + "Are there groups better fit for different lenders?"

**Algorithm Design**:
```
For each customer:
  1. Calculate P(approval | lender A, B, C) using lender-specific models
  2. Get bounty value for each lender (if available, else assume fixed)
  3. Calculate Expected Value = P(approval) × Bounty
  4. Assign customer to lender with max(Expected Value)
  5. If tie, use secondary criteria (response time, relationship targets)
```

**Implementation Considerations**:
- **Capacity Constraints**: Lender volume limits (if applicable)
- **Business Rules**: Minimum FICO requirements, regulatory constraints
- **Fallback Logic**: Second-choice lender if primary rejects
- **Confidence Thresholds**: Only re-route if EV difference > $X

**Deliverables**:
- Matching algorithm pseudocode
- Python implementation
- Unit tests for edge cases
- Algorithm performance benchmarks (latency)

**Tools**: Python (numpy, optimization libraries)

**Validation Test** (Run after completion):
```bash
# Verify subphase 4.2 outputs exist and are correct
python3 tests/test_phase4_matching_algorithm.py

# Expected outputs to verify:
# - src/phase4_revenue_optimization/matching_algorithm.py exists and runs
# - Algorithm assigns each customer to lender with max(Expected Value)
# - Unit tests pass for edge cases (ties, capacity constraints, low confidence)
# - Performance benchmark: <50ms per customer assignment
# - Algorithm handles all 100K customers without errors
# - Output includes confidence scores for assignments
```

#### 4.3 Incremental Revenue Calculation
**Objective**: Quantify revenue lift from optimal matching vs. current state
**Answers**: Q3 - "Calculate how much incremental revenue we could make with optimal matching"

**Methodology**:
1. **Historical Baseline**: Calculate actual revenue from historical data
2. **Optimal Scenario**: Apply matching algorithm to historical data
3. **Revenue Simulation**:
   - For each application, predict optimal lender
   - Use lender-specific model to predict approval probability
   - Calculate expected bounty: `P(approval_optimal) × bounty_optimal`
4. **Incremental Calculation**: `Revenue_optimal - Revenue_baseline`
5. **Confidence Intervals**: Bootstrap resampling (1000 iterations) for CI

**Segment Analysis**:
- Identify customer segments with highest lift potential
- Calculate incremental revenue by segment
- Prioritize segments for A/B testing

**Deliverables**:
- Incremental revenue estimate with 95% confidence interval
- Revenue lift by customer segment
- Sensitivity analysis (bounty variations, model accuracy)
- ROI projection for implementation

**Example Output**:
```
Baseline Revenue:     $2,750,000 (100K apps × $27.50 RPA)
Optimized Revenue:    $3,200,000 (100K apps × $32.00 RPA)
Incremental Revenue:  $450,000 (+16.4%)
Confidence Interval:  [$380K - $520K] (95% CI)
```

**Validation Test** (Run after completion):
```bash
# Verify subphase 4.3 outputs exist and are correct
python3 tests/test_phase4_incremental_revenue.py

# Expected outputs to verify:
# - reports/phase4_revenue_optimization/tables/incremental_revenue_by_segment.csv exists
# - Baseline RPA calculated correctly from historical data
# - Optimized RPA > Baseline RPA (lift > 0%)
# - 95% confidence interval calculated via bootstrap (1000 iterations)
# - Segment-level analysis identifies highest lift opportunities
# - Sensitivity analysis shows impact of ±10% bounty variation
# - ROI projection includes implementation costs
```

#### 4.4 Real-Time Matching Considerations
**Objective**: Design production-ready matching system
**Answers**: Q3 - "What considerations should we have for real-time matching?" (production implementation)

**Technical Requirements**:
1. **Latency**: API response time <200ms (p99)
2. **Availability**: 99.9% uptime with fallback routing
3. **Scalability**: Handle 1000 req/sec peak load
4. **Model Updates**: Hot-swap models without downtime
5. **Monitoring**: Track approval rates, revenue, model drift

**System Design**:
```
User Application
    ↓
Feature Engineering Pipeline (50ms)
    ↓
Parallel Lender Model Scoring (100ms)
    - Model A
    - Model B
    - Model C
    ↓
Matching Algorithm (30ms)
    ↓
Business Rules Validation (20ms)
    ↓
Lender Assignment + API Call
```

**Fallback Logic**:
- If model service down: Use rule-based routing (FICO thresholds)
- If all lenders at capacity: Queue or reject with alternative offer
- If low confidence (<60%): Route to generalist lender (A)

**Deliverables**:
- System architecture diagram
- API specification (OpenAPI/Swagger)
- Load testing plan
- Monitoring and alerting specifications
- Model retraining pipeline design

**Validation Test** (Run after completion):
```bash
# Verify subphase 4.4 outputs exist and are correct
python3 tests/test_phase4_realtime_design.py

# Expected outputs to verify:
# - reports/phase4_revenue_optimization/system_architecture.png exists
# - API specification (OpenAPI YAML) exists and validates
# - Load testing plan documents 1000 req/sec target
# - Monitoring dashboard mockup or specification exists
# - Model retraining pipeline design includes:
#   - Weekly performance monitoring
#   - Monthly retraining schedule
#   - A/B testing protocol for new models
```

#### 4.5 A/B Test Design
**Objective**: Validate algorithm in production with statistical rigor
**Answers**: Q3 - Production validation framework for revenue optimization claims

**Test Design**:
- **Control Group (50%)**: Current routing logic
- **Treatment Group (50%)**: Optimized matching algorithm
- **Randomization**: User-level (consistent experience)
- **Stratification**: Ensure balanced FICO, income, loan amount across groups

**Metrics**:
- **Primary**: Revenue Per Application (RPA)
- **Secondary**:
  - Approval rate
  - Customer satisfaction (NPS)
  - Lender relationship health (volume distribution)
  - Time to decision

**Sample Size Calculation**:
```python
# Assumptions
baseline_rpa = 27.50
expected_lift = 0.15  # 15% increase
std_dev = 50  # estimated
alpha = 0.05
power = 0.80

# Required sample size per group
# Using two-sample t-test formula
# n ≈ 1000-2000 applications per group
```

**Duration**: 2-4 weeks (depends on traffic)

**Success Criteria**:
- RPA lift >10% with p<0.05
- Approval rate maintained or improved
- No degradation in customer satisfaction

**Deliverables**:
- A/B test plan document
- Randomization code
- Statistical analysis plan
- Dashboard for monitoring metrics
- Go/no-go decision framework

**Validation Test** (Run after completion):
```bash
# Verify subphase 4.5 outputs exist and are correct
python3 tests/test_phase4_ab_test_design.py

# Expected outputs to verify:
# - reports/phase4_revenue_optimization/ab_test_plan.pdf exists
# - Randomization code exists and passes unit tests
# - Sample size calculation: n ≈ 1000-2000 per group documented
# - Statistical analysis plan includes:
#   - Primary metric: RPA (two-sample t-test)
#   - Secondary metrics: approval rate, NPS, time to decision
#   - Significance threshold: α = 0.05, power = 0.80
# - Success criteria clearly defined (RPA lift >10%, p<0.05)
# - Go/no-go decision tree documented
```

**UI Development Pattern** (Same as Phases 1, 2 & 3):
Phase 4 follows the same systematic pattern: **Implement → Validate → Display Results**

Each subphase (4.1, 4.2, 4.3, 4.4, 4.5) will have:
1. **Implementation Script**: `src/phase4_revenue_optimization/{subphase}.py`
2. **Validation Test**: `tests/test_phase4_{subphase}.py`
3. **Dashboard Integration**: Added to `ValidationSidebar` under "Phase 4" section
4. **Results Display**: Same `ValidationResults` component shows all outputs

**Dashboard Updates Needed**:
- Add Phase 4 section to sidebar with subphases 4.1, 4.2, 4.3, 4.4, 4.5
- Add Phase 4 subphase tabs to main dashboard
- Results automatically display figures/tables from `reports/phase4_revenue_optimization/`

---

## Technical Architecture

### Technology Stack

**Data Analysis & Modeling**:
- **Language**: Python 3.11+
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Statistics**: scipy, statsmodels
- **Machine Learning**: scikit-learn, xgboost, lightgbm
- **Interpretability**: shap, eli5
- **Notebook Environment**: Jupyter Lab

**Production System** (Post-MVP):
- **API Framework**: FastAPI (Python) or Express.js (Node.js)
- **Model Serving**: TensorFlow Serving, Seldon Core, or custom REST API
- **Database**: PostgreSQL (customer data), Redis (model cache)
- **Message Queue**: RabbitMQ or AWS SQS (async processing)
- **Monitoring**: Prometheus + Grafana, Datadog
- **Cloud**: AWS (ECS/Lambda) or GCP (Cloud Run)

### Project Directory Structure

**Organization Principle**: Structured by analysis phases for easy navigation and progress tracking.

```
rv-case-study/
├── README.md                               # Project overview and setup instructions
├── PRD_LENDING_OPTIMIZATION.md            # This document - analysis roadmap
├── requirements.txt                        # Python dependencies
├── .gitignore                             # Git ignore rules
├── package.json                            # Next.js dependencies
├── next.config.js                          # Next.js configuration
├── tsconfig.json                           # TypeScript configuration
│
├── app/                                    # Next.js App Router (UI + API)
│   ├── layout.tsx                          # Root layout
│   ├── page.tsx                            # Landing page with mode toggle
│   ├── globals.css                         # Global styles
│   │
│   ├── dashboard/                          # Main dashboard pages
│   │   ├── page.tsx                        # Dashboard root (development/results mode)
│   │   ├── layout.tsx                      # Dashboard layout with sidebar
│   │   │
│   │   └── components/                     # Dashboard UI components
│   │       ├── ValidationSidebar.tsx       # Sidebar with run/validate buttons
│   │       ├── ConsoleOutput.tsx           # Real-time console display
│   │       ├── ValidationResults.tsx       # Validation results panel
│   │       ├── ModeToggle.tsx              # Dev ⇄ Results mode toggle
│   │       ├── ProgressTracker.tsx         # Phase completion tracker
│   │       │
│   │       ├── question1/                  # Q1: Variable relationships
│   │       │   ├── FeatureImportanceChart.tsx
│   │       │   ├── FeaturesToDropTable.tsx
│   │       │   └── EngineeredFeaturesCard.tsx
│   │       │
│   │       ├── question2/                  # Q2: Lender analysis
│   │       │   ├── LenderApprovalRatesChart.tsx
│   │       │   ├── LenderPreferenceHeatmap.tsx
│   │       │   └── CustomerSegmentationPlot.tsx
│   │       │
│   │       └── question3/                  # Q3: Revenue optimization
│   │           ├── RevenueBaselineVsOptimizedChart.tsx
│   │           ├── IncrementalRevenueBySegmentTable.tsx
│   │           └── MatchingAlgorithmFlow.tsx
│   │
│   └── api/                                # Next.js API routes
│       ├── analysis/
│       │   └── run/
│       │       └── route.ts                # POST /api/analysis/run
│       ├── validation/
│       │   └── run/
│       │       └── route.ts                # POST /api/validation/run
│       └── outputs/
│           ├── list/
│           │   └── route.ts                # GET /api/outputs/list
│           └── [type]/
│               └── [filename]/
│                   └── route.ts            # GET /api/outputs/{tables|figures}/{file}
│
├── lib/                                    # Shared utilities
│   ├── python-executor.ts                  # Execute Python scripts
│   ├── validation-parser.ts               # Parse validation JSON output
│   └── file-reader.ts                      # Read analysis output files
│
├── data/                                   # Data storage (all phases)
│   ├── raw/
│   │   └── Pre-Super_Day_candidate_dataset.xlsx  # Original dataset
│   ├── processed/
│   │   ├── cleaned_data.csv               # After Phase 1.3 (missing values handled)
│   │   ├── features_engineered.csv        # After Phase 1.2 (derived features)
│   │   ├── train.csv                      # Training set (70%)
│   │   └── test.csv                       # Test set (30%)
│   └── outputs/
│       ├── phase2_feature_importance.csv  # Phase 2 results
│       ├── phase3_lender_predictions.csv  # Phase 3 model outputs
│       └── phase4_revenue_simulation.csv  # Phase 4 optimization results
│
├── notebooks/                              # Jupyter notebooks by phase
│   ├── phase1_eda/
│   │   ├── 1.1_univariate_analysis.ipynb  # Q1: Variable distributions
│   │   ├── 1.2_bivariate_analysis.ipynb   # Q1: Feature engineering
│   │   └── 1.3_missing_values.ipynb       # Data quality
│   ├── phase2_feature_importance/
│   │   ├── 2.1_statistical_importance.ipynb  # Q1: Statistical rankings
│   │   ├── 2.2_ml_feature_importance.ipynb   # Q1: ML-based rankings
│   │   └── 2.3_feature_validation.ipynb      # Q1: Engineered feature testing
│   ├── phase3_lender_analysis/
│   │   ├── 3.1_lender_profiling.ipynb     # Q2: Approval rates by lender
│   │   ├── 3.2_lender_models.ipynb        # Q2: Lender-specific models
│   │   └── 3.3_lender_specialization.ipynb # Q2: Lender differences
│   ├── phase4_revenue_optimization/
│   │   ├── 4.1_current_state_revenue.ipynb    # Q3: Baseline RPA
│   │   ├── 4.2_matching_algorithm.ipynb       # Q3: Optimal matching
│   │   ├── 4.3_incremental_revenue.ipynb      # Q3: Revenue lift calculation
│   │   ├── 4.4_realtime_design.ipynb          # Q3: Production considerations
│   │   └── 4.5_ab_test_design.ipynb           # Q3: Validation framework
│   └── final_report.ipynb                 # Executive summary (all questions)
│
├── src/                                    # Python source code organized by phase
│   ├── __init__.py
│   │
│   ├── phase1_eda/                        # Phase 1: EDA utilities
│   │   ├── __init__.py
│   │   ├── data_loader.py                 # Load and validate raw data
│   │   ├── univariate.py                  # Single variable analysis
│   │   ├── bivariate.py                   # Variable relationships
│   │   ├── feature_engineering.py         # DTI, LTI, FICO bins, etc.
│   │   └── missing_values.py              # Handle Employment_Sector nulls
│   │
│   ├── phase2_feature_importance/         # Phase 2: Feature selection
│   │   ├── __init__.py
│   │   ├── statistical_tests.py           # Mutual info, ANOVA, chi-square
│   │   ├── ml_importance.py               # Random Forest, XGBoost importance
│   │   ├── shap_analysis.py               # SHAP value calculations
│   │   └── feature_validation.py          # Baseline vs engineered comparison
│   │
│   ├── phase3_lender_analysis/            # Phase 3: Lender-specific analysis
│   │   ├── __init__.py
│   │   ├── lender_profiling.py            # Approval rates by segment
│   │   ├── lender_models.py               # Train 3 separate models
│   │   ├── model_evaluation.py            # AUC-ROC, calibration, metrics
│   │   └── specialization_analysis.py     # ANOVA, clustering, segments
│   │
│   ├── phase4_revenue_optimization/       # Phase 4: Matching & revenue
│   │   ├── __init__.py
│   │   ├── revenue_calculator.py          # Current state RPA calculations
│   │   ├── matching_algorithm.py          # Optimal lender assignment
│   │   ├── simulation.py                  # Monte Carlo revenue simulation
│   │   └── ab_test_design.py              # Sample size, power calculations
│   │
│   └── utils/                             # Shared utilities (all phases)
│       ├── __init__.py
│       ├── config.py                      # Configuration management
│       ├── visualization.py               # Plotting functions
│       ├── metrics.py                     # Custom evaluation metrics
│       └── logger.py                      # Logging setup
│
├── models/                                 # Trained model artifacts (Phase 3+)
│   ├── phase3_lender_models/
│   │   ├── lender_a_model.pkl             # Lender A XGBoost model
│   │   ├── lender_b_model.pkl             # Lender B XGBoost model
│   │   ├── lender_c_model.pkl             # Lender C XGBoost model
│   │   └── feature_scaler.pkl             # StandardScaler for features
│   └── model_metadata.json                # Model versions, performance metrics
│
├── tests/                                  # Validation tests for all phases
│   ├── test_phase1_foundation.py          # Validate data loading (Phase 1 foundation)
│   ├── test_phase1_univariate.py          # Validate Phase 1.1 outputs
│   ├── test_phase1_bivariate.py           # Validate Phase 1.2 outputs
│   ├── test_phase1_missing_values.py      # Validate Phase 1.3 outputs
│   ├── test_phase2_statistical.py         # Validate Phase 2.1 outputs
│   ├── test_phase2_ml_importance.py       # Validate Phase 2.2 outputs
│   ├── test_phase2_validation.py          # Validate Phase 2.3 outputs
│   ├── test_phase3_profiling.py           # Validate Phase 3.1 outputs
│   ├── test_phase3_lender_models.py       # Validate Phase 3.2 outputs
│   ├── test_phase3_specialization.py      # Validate Phase 3.3 outputs
│   ├── test_phase4_baseline_revenue.py    # Validate Phase 4.1 outputs
│   ├── test_phase4_matching_algorithm.py  # Validate Phase 4.2 outputs
│   ├── test_phase4_incremental_revenue.py # Validate Phase 4.3 outputs
│   ├── test_phase4_realtime.py            # Validate Phase 4.4 outputs
│   └── test_phase4_ab_test_design.py      # Validate Phase 4.5 outputs
│
├── documentation/                          # Planning and guide documents
│   ├── PHASE1_GUIDE.md                    # Phase 1 implementation guide
│   ├── PHASE2_GUIDE.md                    # Phase 2 implementation guide
│   ├── PHASE3_GUIDE.md                    # Phase 3 implementation guide
│   └── PHASE4_GUIDE.md                    # Phase 4 implementation guide
│
├── reports/                                # Analysis outputs by phase
│   ├── phase1_eda/
│   │   ├── figures/                       # EDA plots
│   │   │   ├── approval_by_fico.png
│   │   │   ├── income_distribution.png
│   │   │   └── correlation_heatmap.png
│   │   └── tables/
│   │       ├── approval_rates_by_category.csv
│   │       └── summary_statistics.csv
│   │
│   ├── phase2_feature_importance/
│   │   ├── figures/
│   │   │   ├── feature_importance_rf.png
│   │   │   ├── shap_summary.png
│   │   │   └── feature_comparison.png
│   │   └── tables/
│   │       ├── feature_rankings.csv
│   │       └── features_to_drop.csv
│   │
│   ├── phase3_lender_analysis/
│   │   ├── figures/
│   │   │   ├── lender_a_profile.png
│   │   │   ├── lender_b_profile.png
│   │   │   ├── lender_c_profile.png
│   │   │   └── lender_comparison.png
│   │   └── tables/
│   │       ├── lender_approval_rates.csv
│   │       ├── model_performance.csv
│   │       └── lender_specialization.csv
│   │
│   ├── phase4_revenue_optimization/
│   │   ├── figures/
│   │   │   ├── revenue_by_segment.png
│   │   │   ├── optimal_matching_flow.png
│   │   │   └── incremental_revenue.png
│   │   └── tables/
│   │       ├── baseline_rpa.csv
│   │       ├── optimized_rpa.csv
│   │       └── incremental_revenue_by_segment.csv
│   │
│   └── final_report/
│       ├── executive_summary.pdf          # Final presentation
│       ├── technical_documentation.pdf    # Detailed methodology
│       └── business_recommendations.pdf   # Action items
│
├── tests/                                  # Unit tests by phase
│   ├── __init__.py
│   ├── test_phase1_eda.py
│   ├── test_phase2_feature_importance.py
│   ├── test_phase3_lender_models.py
│   └── test_phase4_optimization.py
│
├── config/                                 # Configuration files
│   ├── data_config.yaml                   # Data paths, column definitions
│   ├── model_config.yaml                  # XGBoost hyperparameters
│   └── feature_config.yaml                # Feature engineering specs
│
└── api/                                    # Production API (Phase 4.4 output)
    ├── app.py                             # FastAPI application
    ├── models.py                          # Pydantic data models
    ├── routes.py                          # API endpoints
    ├── matching_service.py                # Real-time matching logic
    └── requirements_api.txt               # API-specific dependencies
```

**Directory Usage Guidelines**:
- **Sequential Analysis**: Work through `notebooks/phase1_eda/` → `phase2_feature_importance/` → `phase3_lender_analysis/` → `phase4_revenue_optimization/`
- **Code Reuse**: Import functions from `src/phaseX_*/` into notebooks for clean, reproducible analysis
- **Output Organization**: Save all plots to `reports/phaseX_*/figures/`, all tables to `reports/phaseX_*/tables/`
- **Model Persistence**: Save trained models to `models/phase3_lender_models/` with version tracking
- **Question Tracking**: Notebooks clearly labeled with question numbers (Q1, Q2, Q3) for easy reference

---

## UI/Dashboard Design Guidance

**Purpose**: Interactive dashboard with **run & validate** capabilities for each subphase, presenting analysis findings organized by research questions

### Architecture Overview

**Dual-Mode Dashboard**:
1. **Development/Validation Mode**: Run analysis scripts and validate outputs per subphase
2. **Results/Presentation Mode**: View finalized analysis findings organized by research questions

### Layout Architecture

**Single-Page Dashboard** with integrated validation controls:

```
Dashboard Structure:
├── Header: Project title + progress tracker (subphases completed)
├── Mode Toggle: Development Mode ⇄ Results Mode
├── ValidationSidebar (Development Mode):
│   ├── Phase 1: EDA
│   │   ├── [▶ Run 1.1] Univariate → [✓ Validate] → Status: ✅/❌
│   │   ├── [▶ Run 1.2] Bivariate → [✓ Validate] → Status: ⏳
│   │   └── [▶ Run 1.3] Missing Values → [✓ Validate] → Status: ⏳
│   ├── Phase 2: Feature Importance
│   │   ├── [▶ Run 2.1] Statistical → [✓ Validate] → Status: ⏳
│   │   ├── [▶ Run 2.2] ML Importance → [✓ Validate] → Status: ⏳
│   │   └── [▶ Run 2.3] Validation → [✓ Validate] → Status: ⏳
│   ├── Phase 3: Lender Analysis
│   │   ├── [▶ Run 3.1] Profiling → [✓ Validate] → Status: ⏳
│   │   ├── [▶ Run 3.2] Models → [✓ Validate] → Status: ⏳
│   │   └── [▶ Run 3.3] Specialization → [✓ Validate] → Status: ⏳
│   └── Phase 4: Revenue Optimization
│       ├── [▶ Run 4.1] Baseline → [✓ Validate] → Status: ⏳
│       ├── [▶ Run 4.2] Algorithm → [✓ Validate] → Status: ⏳
│       ├── [▶ Run 4.3] Incremental → [✓ Validate] → Status: ⏳
│       ├── [▶ Run 4.4] Real-time → [✓ Validate] → Status: ⏳
│       └── [▶ Run 4.5] A/B Test → [✓ Validate] → Status: ⏳
│
└── MainContent:
    ├── Development Mode:
    │   ├── Console Output Window (real-time logs)
    │   ├── Validation Results Panel (checks passed/failed)
    │   └── Generated Outputs Preview (tables, figures)
    │
    └── Results Mode (after validation passes):
        ├── Question 1: Variable Relationships (Phases 1-2)
        ├── Question 2: Lender Approval Rates (Phase 3)
        └── Question 3: Revenue Optimization (Phase 4)
```

### Development Mode Features

**Run Button Functionality**:
```typescript
// When user clicks "Run 1.1" button
onClick={() => {
  // 1. Show loading state
  setStatus('running')

  // 2. Call API to execute Python script
  const result = await fetch('/api/analysis/run', {
    method: 'POST',
    body: JSON.stringify({
      phase: 'phase1_eda',
      subphase: 'univariate',
      script: 'src/phase1_eda/univariate.py'
    })
  })

  // 3. Stream console output to UI in real-time
  streamConsoleOutput(result.logs)

  // 4. Update status based on result
  setStatus(result.success ? 'completed' : 'failed')
}}
```

**Validate Button Functionality**:
```typescript
// When user clicks "Validate" button
onClick={async () => {
  // 1. Show validation running state
  setValidationStatus('validating')

  // 2. Call API to run validation test
  const result = await fetch('/api/validation/run', {
    method: 'POST',
    body: JSON.stringify({
      test: 'tests/test_phase1_univariate.py'
    })
  })

  // 3. Display validation results
  setValidationResults({
    passed: result.checks_passed,
    total: result.total_checks,
    passRate: result.pass_rate,
    details: result.check_details
  })

  // 4. Update status badge
  setValidationStatus(result.pass_rate >= 0.80 ? 'passed' : 'failed')
}}
```

**Status Indicators**:
- ⏳ **Pending**: Not yet run
- 🔄 **Running**: Script currently executing
- ✅ **Passed**: Validation passed (≥80% checks)
- ❌ **Failed**: Validation failed (<80% checks)
- ⚠️ **Warning**: Completed but with warnings

### Validation Results Panel

```typescript
interface ValidationResult {
  subphase: string            // "Phase 1.1: Univariate Analysis"
  checksPassedCount: number   // 19
  totalChecksCount: number    // 19
  passRate: number           // 1.00
  status: 'passed' | 'failed' | 'warning'
  details: {
    category: string         // "Tables", "Figures", "Content"
    checks: Array<{
      name: string          // "FICO bins table exists"
      passed: boolean       // true
      message: string       // "✅ Found: approval_by_fico_bins.csv"
    }>
  }[]
  consoleOutput: string      // Full console output from script
  generatedFiles: string[]   // List of files created
  insights: string[]         // Key insights discovered
}
```

**UI Display**:
```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1.1: Univariate Analysis - Validation Results        │
├─────────────────────────────────────────────────────────────┤
│ Status: ✅ PASSED (19/19 checks, 100%)                      │
│                                                             │
│ ✅ Tables (5/5)                                             │
│   ✓ approval_by_reason.csv                                 │
│   ✓ approval_by_employment_status.csv                      │
│   ✓ approval_by_lender.csv                                 │
│   ✓ approval_by_fico_bins.csv                              │
│   ✓ numerical_variable_statistics.csv                      │
│                                                             │
│ ✅ Figures (8/8)                                            │
│   ✓ approval_by_reason.png                                 │
│   ✓ fico_score_by_approval.png                             │
│   ... (6 more)                                             │
│                                                             │
│ ✅ Content Quality (6/6)                                    │
│   ✓ FICO bins show 43.0% variation                         │
│   ✓ All 3 lenders analyzed (A, B, C)                       │
│   ✓ Meaningful insights generated                          │
│                                                             │
│ 💡 Key Insights:                                            │
│   • FICO 800+: 45.8% approval vs <580: 2.8%               │
│   • Lender C most lenient (17.1% approval)                 │
│   • Employment status: 2.2x difference                      │
│                                                             │
│ [View Console Output] [View Generated Files]               │
└─────────────────────────────────────────────────────────────┘
```

### Backend API Architecture

**API Routes** (`app/api/`):

```typescript
// app/api/analysis/run/route.ts
POST /api/analysis/run
Request: {
  phase: string,        // "phase1_eda"
  subphase: string,     // "univariate"
  script: string        // "src/phase1_eda/univariate.py"
}
Response: {
  success: boolean,
  logs: string[],       // Real-time console output
  executionTime: number,
  error?: string
}

// app/api/validation/run/route.ts
POST /api/validation/run
Request: {
  test: string         // "tests/test_phase1_univariate.py"
}
Response: {
  success: boolean,
  checks_passed: number,
  total_checks: number,
  pass_rate: number,
  check_details: Array<{
    category: string,
    checks: Array<{
      name: string,
      passed: boolean,
      message: string
    }>
  }>,
  insights: string[],
  generatedFiles: string[]
}

// app/api/outputs/list/route.ts
GET /api/outputs/list?subphase=phase1_univariate
Response: {
  tables: string[],     // List of CSV files
  figures: string[],    // List of PNG files
  metadata: {
    createdAt: string,
    fileCount: number
  }
}

// app/api/outputs/[type]/[filename]/route.ts
GET /api/outputs/tables/approval_by_fico_bins.csv
GET /api/outputs/figures/approval_by_fico_bins.png
Response: File stream
```

**Python Script Output Format**:

All Python analysis scripts must output **JSON-formatted results** to stdout for UI parsing:

```python
# src/phase1_eda/univariate.py - Updated output format

import json
import sys

def generate_univariate_summary(df, output_dir='reports/phase1_eda'):
    # ... existing analysis code ...

    # Build structured output for UI
    output = {
        "success": True,
        "subphase": "Phase 1.1: Univariate Analysis",
        "summary": {
            "overall_approval_rate": overall_approval,
            "variables_analyzed": {
                "categorical": len(categorical_vars),
                "numerical": len(numerical_vars)
            }
        },
        "insights": [
            f"FICO 800+: {best_fico_rate:.1%} approval vs <580: {worst_fico_rate:.1%}",
            f"Lender C most lenient ({lender_c_rate:.1%} approval)",
            f"Employment: {best_emp_rate:.1%} vs {worst_emp_rate:.1%} ({diff:.1f}x difference)"
        ],
        "outputs": {
            "tables": [
                "approval_by_reason.csv",
                "approval_by_employment_status.csv",
                "approval_by_lender.csv",
                "approval_by_fico_bins.csv",
                "numerical_variable_statistics.csv"
            ],
            "figures": [
                "approval_by_reason.png",
                "approval_by_employment_status.png",
                "approval_by_lender.png",
                "approval_by_fico_bins.png",
                "fico_score_by_approval.png",
                "loan_amount_by_approval.png",
                "monthly_gross_income_by_approval.png",
                "monthly_housing_payment_by_approval.png"
            ]
        },
        "execution_time": time.time() - start_time
    }

    # Output JSON to stdout (last line)
    print("\n__JSON_OUTPUT__")
    print(json.dumps(output, indent=2))

    return results
```

**Validation Test Output Format**:

```python
# tests/test_phase1_univariate.py - Updated output format

def validate_phase1_univariate():
    # ... existing validation code ...

    # Build structured validation results
    validation_output = {
        "success": passed_checks >= total_checks * 0.8,
        "subphase": "Phase 1.1: Univariate Analysis",
        "checks_passed": passed_checks,
        "total_checks": total_checks,
        "pass_rate": pass_rate,
        "details": [
            {
                "category": "Tables",
                "checks": [
                    {"name": "approval_by_reason.csv", "passed": True, "message": "✅ Found"},
                    {"name": "approval_by_lender.csv", "passed": True, "message": "✅ Found"},
                    # ... more checks
                ]
            },
            {
                "category": "Figures",
                "checks": [
                    {"name": "approval_by_fico_bins.png", "passed": True, "message": "✅ Found"},
                    # ... more checks
                ]
            },
            {
                "category": "Content Quality",
                "checks": [
                    {"name": "FICO variation", "passed": True, "message": "✅ 43.0% range"},
                    {"name": "Lender coverage", "passed": True, "message": "✅ All 3 lenders"},
                    # ... more checks
                ]
            }
        ],
        "insights": [
            "FICO 800+: 45.8% approval vs <580: 2.8%",
            "Lender C most lenient (17.1% approval)",
            "Employment status: 2.2x difference"
        ]
    }

    # Output JSON to stdout (last line)
    print("\n__JSON_OUTPUT__")
    print(json.dumps(validation_output, indent=2))

    return validation_output["success"]
```

### Component Organization by Question

#### Question 1: Variable Relationship with Approvability
**Section ID**: `#question-1`
**Components**:
- `FeatureImportanceChart`: Top 10 most predictive variables (bar chart)
- `FeaturesToDropTable`: Variables with p>0.05 or low importance
- `EngineeredFeaturesCard`: DTI, LTI with AUC lift metrics
- `FeatureComparisonChart`: Baseline vs. engineered feature performance

**Key Metrics Displayed**:
- Top 5 predictive variables with importance scores
- Number of features recommended for removal
- AUC improvement from feature engineering

#### Question 2: Lender Approval Rates
**Section ID**: `#question-2`
**Components**:
- `LenderApprovalRatesChart`: Approval rates by lender (A: 10.97%, B: 7.13%, C: 17.06%)
- `LenderComparisonTable`: Side-by-side comparison of lender preferences
- `LenderSpecializationHeatmap`: Approval rates by lender × customer segment
- `LenderFeatureImportance`: Which features matter most per lender

**Key Metrics Displayed**:
- Approval rate by lender with confidence intervals
- Statistical significance of lender differences (p-values)
- "Sweet spot" customer profiles per lender

#### Question 3: Revenue Optimization
**Section ID**: `#question-3`
**Components**:
- `CurrentVsOptimizedRPA`: Before/after revenue comparison (gauge charts)
- `IncrementalRevenueCard`: Dollar amount + percentage lift with CI
- `CustomerSegmentReallocation`: Flow diagram showing re-routing opportunities
- `MatchingAlgorithmFlow`: Visual explanation of optimal assignment logic
- `ABTestDesign`: Sample size, duration, success criteria

**Key Metrics Displayed**:
- Baseline RPA vs. Optimized RPA
- Incremental revenue: $XXX,XXX (+XX%)
- Customer segments with highest re-routing potential
- Expected implementation timeline

### Navigation & User Experience

**Sticky Question Navigator** (Left Sidebar):
```tsx
<QuestionNav>
  <NavItem href="#question-1" status="completed">
    Q1: Variables
  </NavItem>
  <NavItem href="#question-2" status="completed">
    Q2: Lenders
  </NavItem>
  <NavItem href="#question-3" status="completed">
    Q3: Revenue
  </NavItem>
</QuestionNav>
```

**Interaction Patterns**:
- Click nav item → smooth scroll to question section
- Scroll page → auto-highlight active nav item
- Each section has "Back to Top" button
- Export buttons per section (CSV for tables, PNG for charts)

### Technical Implementation

**Framework**: Next.js 15 + React 19 + TypeScript + Tailwind CSS

**Component Structure**:
```
app/
├── dashboard/
│   ├── page.tsx                           # Main dashboard page
│   ├── layout.tsx                         # Dashboard layout with nav
│   └── components/
│       ├── QuestionNav.tsx                # Sticky sidebar navigation
│       │
│       ├── question1/                     # Q1 Components
│       │   ├── Question1Results.tsx       # Container
│       │   ├── FeatureImportanceChart.tsx
│       │   ├── FeaturesToDropTable.tsx
│       │   └── EngineeredFeaturesCard.tsx
│       │
│       ├── question2/                     # Q2 Components
│       │   ├── Question2Results.tsx       # Container
│       │   ├── LenderApprovalRatesChart.tsx
│       │   ├── LenderComparisonTable.tsx
│       │   ├── LenderSpecializationHeatmap.tsx
│       │   └── LenderFeatureImportance.tsx
│       │
│       └── question3/                     # Q3 Components
│           ├── Question3Results.tsx       # Container
│           ├── CurrentVsOptimizedRPA.tsx
│           ├── IncrementalRevenueCard.tsx
│           ├── CustomerSegmentReallocation.tsx
│           ├── MatchingAlgorithmFlow.tsx
│           └── ABTestDesign.tsx
```

**Data Flow**:
```
Analysis Results (CSV/JSON from notebooks)
    ↓
API Routes (/api/results/q1, /api/results/q2, /api/results/q3)
    ↓
Dashboard Components (fetch data on mount)
    ↓
Visualization Libraries (recharts, d3, plotly)
```

**State Management**:
- **Client-side**: Zustand for dashboard filters, active question, chart interactions
- **Server-side**: Next.js API routes fetch pre-computed results from analysis outputs

### Data Sources for Dashboard

Each question pulls from analysis outputs:

**Question 1**:
- `data/outputs/phase2_feature_importance.csv`
- `reports/phase2_feature_importance/tables/feature_rankings.csv`
- `reports/phase2_feature_importance/tables/features_to_drop.csv`

**Question 2**:
- `reports/phase3_lender_analysis/tables/lender_approval_rates.csv`
- `reports/phase3_lender_analysis/tables/model_performance.csv`
- `reports/phase3_lender_analysis/tables/lender_specialization.csv`

**Question 3**:
- `data/outputs/phase4_revenue_simulation.csv`
- `reports/phase4_revenue_optimization/tables/baseline_rpa.csv`
- `reports/phase4_revenue_optimization/tables/incremental_revenue_by_segment.csv`

### Design Principles

1. **Question-Centric**: Each section clearly labeled with original question text
2. **Progressive Disclosure**: Start with high-level metrics, expand for details
3. **Visual Hierarchy**: Charts first, tables second, technical details collapsible
4. **Actionability**: Clear recommendations in each section
5. **Export-Friendly**: All data tables and charts downloadable for presentations

### Accessibility & Responsiveness

- **Responsive**: Desktop-first (1920x1080), tablet support (1024x768+)
- **Keyboard Navigation**: Tab through sections, Enter to expand details
- **Screen Reader**: Semantic HTML, ARIA labels for charts
- **Print-Friendly**: CSS print styles for report generation

### Future Enhancements (Post-MVP)

- **Interactive Filtering**: Filter results by FICO range, income quartile, lender
- **What-If Scenarios**: Adjust bounty values, see real-time revenue impact
- **Real-Time Monitoring**: Once production API deployed, show live metrics
- **Historical Comparison**: Track metrics over time as models retrain

### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────┐
│  DATA INGESTION                                         │
│  - Load Excel file                                      │
│  - Validate schema                                      │
│  - Handle missing values                                │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  FEATURE ENGINEERING                                    │
│  - Debt-to-Income Ratio                                │
│  - Loan-to-Income Ratio                                │
│  - Income Quartiles                                     │
│  - Custom FICO Bins                                     │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  EXPLORATORY ANALYSIS                                   │
│  - Univariate distributions                            │
│  - Correlation analysis                                │
│  - Statistical tests                                    │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  MODEL TRAINING                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Lender A   │  │  Lender B   │  │  Lender C   │    │
│  │   Model     │  │   Model     │  │   Model     │    │
│  └─────────────┘  └─────────────┘  └─────────────┘    │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  OPTIMIZATION ALGORITHM                                 │
│  - Predict approval probability per lender             │
│  - Calculate expected value                            │
│  - Assign optimal lender                               │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  REVENUE SIMULATION                                     │
│  - Current state RPA                                   │
│  - Optimized state RPA                                 │
│  - Incremental revenue calculation                     │
└─────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────┐
│  BUSINESS RECOMMENDATIONS                               │
│  - Feature collection priorities                       │
│  - Lender relationship strategy                        │
│  - A/B test design                                     │
│  - Implementation roadmap                              │
└─────────────────────────────────────────────────────────┘
```

---

## Expected Deliverables

### Week 1: Exploratory Analysis
1. **EDA Report** (Jupyter notebook + PDF)
   - Variable distributions and approval rates
   - Correlation analysis
   - Statistical test results
   - Data quality assessment

2. **Feature Engineering Specifications**
   - Derived feature definitions
   - Transformation logic
   - Code for implementation

3. **Initial Insights Memo**
   - Top 5 predictive variables (preliminary)
   - Observed lender differences
   - Recommended features to drop

### Week 2: Feature Importance & Lender Profiling
4. **Feature Importance Report**
   - Statistical rankings (mutual information, ANOVA, chi-square)
   - ML model feature importance (Random Forest, XGBoost)
   - SHAP value analysis
   - Final feature recommendations

5. **Lender Profile Reports** (3 separate reports)
   - Approval rate by segment
   - Statistical comparisons (approved vs. denied)
   - Identified thresholds
   - Lender specialization summary

6. **Lender-Specific Models** (3 trained models)
   - Model performance metrics
   - Feature importance per lender
   - Calibrated probability outputs

### Week 3-4: Revenue Optimization
7. **Current State Revenue Analysis**
   - Baseline RPA calculations
   - Revenue by lender and segment
   - Bounty structure documentation

8. **Optimal Matching Algorithm**
   - Algorithm design document
   - Python implementation
   - Unit tests
   - Performance benchmarks

9. **Incremental Revenue Report**
   - Revenue lift estimate with confidence interval
   - Segment-level analysis
   - Sensitivity analysis
   - ROI projection

10. **Real-Time System Design**
    - Architecture diagram
    - API specifications
    - Monitoring plan
    - A/B test design

### Final Deliverables
11. **Executive Summary** (PowerPoint/PDF)
    - Key findings (3-5 slides)
    - Recommended actions
    - Expected revenue impact
    - Implementation timeline

12. **Technical Documentation**
    - Code repository (GitHub)
    - Model cards (model performance, limitations)
    - Feature documentation
    - API documentation (if applicable)

13. **Business Recommendations Document**
    - Feature collection priorities
    - Lender relationship strategy
    - Customer segmentation strategy
    - Phase 1 implementation plan

---

## Success Metrics

### Analysis Success Criteria
- **Feature Identification**: Identify top 5 predictive variables with statistical significance (p<0.05)
- **Model Performance**: Lender-specific models achieve >70% AUC-ROC on test set
- **Lender Differentiation**: Demonstrate statistically significant differences between lenders (p<0.01)
- **Revenue Opportunity**: Quantify incremental revenue with ±10% confidence interval

### Business Impact Targets (Post-Implementation)
- **Revenue Per Application**: Increase RPA by >10% ($2.75+)
- **Approval Rate**: Maintain or improve overall approval rate (>10.98%)
- **Lender Relationship Health**: Balanced volume distribution aligned with capacity
- **Customer Experience**: Reduce rejection rate for qualified customers by >15%

### Technical Performance Targets (Production)
- **API Latency**: <200ms p99 response time
- **Availability**: 99.9% uptime
- **Model Accuracy**: Maintain >65% precision on approval predictions
- **System Scalability**: Handle 1000 req/sec peak load

---

## Implementation Timeline

### Phase 1: Analysis & Modeling (4 weeks)
**Week 1: Exploratory Analysis**
- Days 1-2: Data loading, cleaning, EDA
- Days 3-4: Feature engineering and validation
- Day 5: Initial insights memo

**Week 2: Feature Importance & Lender Profiling**
- Days 1-2: Statistical feature importance
- Days 3-4: Lender-specific analysis
- Day 5: Feature and lender profile reports

**Week 3: Model Training**
- Days 1-3: Train and tune lender-specific models
- Days 4-5: Model evaluation and calibration

**Week 4: Revenue Optimization**
- Days 1-2: Matching algorithm development
- Days 3-4: Revenue simulation and incremental calculation
- Day 5: Executive summary and recommendations

### Phase 2: Production Implementation (8-12 weeks, post-analysis)
**Weeks 5-6: System Design**
- API architecture design
- Database schema design
- Infrastructure provisioning

**Weeks 7-10: Development**
- API implementation
- Model serving integration
- Business rules engine
- Monitoring and logging

**Weeks 11-12: Testing & Deployment**
- Unit and integration testing
- Load testing
- Staging deployment
- A/B test launch

**Weeks 13-16: A/B Testing & Iteration**
- Monitor metrics
- Statistical analysis
- Model refinement
- Full rollout or rollback decision

---

## Technical Risks & Mitigations

### Risk 1: Model Overfitting
**Risk**: Lender-specific models overfit to training data, poor generalization

**Indicators**:
- Large gap between train and test AUC (>10%)
- High variance in cross-validation scores
- Poor performance on recent data

**Mitigation**:
- Use 5-fold cross-validation during training
- Regularize models (L2 penalty, max_depth limits)
- Monitor learning curves for overfitting signs
- Validate on time-based holdout (most recent 20% of data)
- Use ensemble methods (Random Forest, XGBoost) for robustness

### Risk 2: Data Leakage
**Risk**: Future information leaks into features, inflating model performance

**Potential Sources**:
- `bounty` variable available at prediction time (should be excluded)
- `applications` count may leak approval likelihood
- Temporal ordering issues

**Mitigation**:
- Strict feature audit before modeling
- Remove `bounty` and `applications` from training features
- Use only information available at application time
- Time-based train/test split (train on older data, test on newer)

### Risk 3: Class Imbalance
**Risk**: Only 10.98% approval rate creates severe class imbalance

**Impact**:
- Models may predict "deny" for all cases (89% accuracy but useless)
- Poor precision and recall on approval class

**Mitigation**:
- Use stratified sampling in train/test split
- Apply SMOTE (Synthetic Minority Over-sampling) or class weights
- Optimize for F1-score or AUC-ROC, not accuracy
- Use precision-recall curves for evaluation
- Consider ensemble methods with balanced subsamples

### Risk 4: Lender Volume Constraints
**Risk**: Optimal matching sends too many customers to high-approval lender (C)

**Impact**:
- Lender C capacity exceeded
- Need fallback routing logic
- Business relationships with A/B strained

**Mitigation**:
- Implement capacity constraints in matching algorithm
- Use round-robin or weighted routing when capacity full
- Monitor lender volume distribution in A/B test
- Negotiate volume flexibility with lenders before rollout

### Risk 5: Bounty Structure Unknown
**Risk**: Assumption that bounty is fixed per lender may be incorrect

**Indicators**:
- Bounty varies by customer or loan characteristics
- Lender-specific bounty rates not documented

**Mitigation**:
- Analyze bounty distribution by lender and customer segment (Phase 4.1)
- If varies: Incorporate bounty prediction into algorithm
- If fixed: Document assumptions clearly
- Validate with business stakeholders early

### Risk 6: Real-Time Latency Requirements
**Risk**: Model inference + matching algorithm exceeds acceptable latency (<200ms)

**Impact**:
- Poor user experience
- Increased infrastructure costs
- Need to simplify algorithm

**Mitigation**:
- Profile code for bottlenecks
- Use model quantization or distillation
- Cache frequent feature computations
- Pre-compute customer embeddings if possible
- Consider rule-based fallback for high-load periods

### Risk 7: Model Staleness
**Risk**: Customer and lender behavior changes over time, models become stale

**Indicators**:
- Declining approval prediction accuracy
- Drift in feature distributions
- Business policy changes at lenders

**Mitigation**:
- Monitor model performance metrics weekly
- Detect feature drift (Kolmogorov-Smirnov test)
- Implement automated retraining pipeline (monthly)
- A/B test model updates before full deployment
- Maintain champion-challenger model framework

---

## Assumptions & Constraints

### Assumptions
1. **Historical Data Representativeness**: Past patterns will hold in near future
2. **Lender Stability**: Lender approval criteria have not changed significantly recently
3. **Bounty Structure**: Bounty is fixed per lender or varies predictably
4. **Feature Availability**: All features available at prediction time in production
5. **Independent Applications**: Applications are independent (no multi-lender submissions)
6. **Honest Reporting**: Data accurately reflects actual approvals and revenue

### Constraints
1. **Data Limitations**: Single historical dataset, no live traffic for testing
2. **Computational Resources**: Analysis on local machine or single cloud instance
3. **Timeline**: 4 weeks for analysis, 8-12 weeks for production implementation
4. **Privacy**: Must anonymize data in reports, no PII in code repository
5. **Business Rules**: Must respect lender-specific compliance requirements
6. **Capacity**: Lender volume limits may restrict optimal routing

---

## Open Questions

### Business Questions
1. **Bounty Variation**: Does bounty vary by customer, loan amount, or lender? If so, how?
2. **Lender Capacity**: What are volume limits per lender? Are they negotiable?
3. **Approval Timing**: How quickly do lenders respond? Is there SLA variance?
4. **Customer Lifetime Value**: Should we optimize for first-loan revenue or long-term CLV?
5. **Relationship Targets**: Are there contractual volume commitments with lenders?

### Technical Questions
6. **Real-Time Requirements**: What is acceptable latency for lender assignment?
7. **Fallback Behavior**: If optimal lender rejects, can we re-route to second choice?
8. **Model Updates**: How frequently should models be retrained? Approval process?
9. **Infrastructure**: Where will production system be hosted (AWS, GCP, on-prem)?
10. **Monitoring**: What alerts are critical (approval rate drop, revenue decline, latency spike)?

### Data Questions
11. **Missing Sector**: Is Employment_Sector missing at random or systematically (unemployment)?
12. **FICO Source**: What credit bureau provides FICO scores? Are they consistent?
13. **Income Verification**: Is Monthly_Gross_Income self-reported or verified?
14. **Temporal Patterns**: Are there seasonality effects (holidays, economic cycles)?
15. **Duplicate Applications**: Can same User ID appear multiple times? If so, how to handle?

---

## Appendix

### A. Statistical Methods Reference

**Chi-Square Test for Independence**:
- Purpose: Test if categorical variable is associated with approval
- Null Hypothesis: Variable is independent of approval
- Interpretation: p<0.05 indicates significant association

**ANOVA (Analysis of Variance)**:
- Purpose: Test if mean of numerical variable differs between approved/denied
- Null Hypothesis: Means are equal across groups
- Interpretation: p<0.05 indicates significant difference

**Mutual Information**:
- Purpose: Measure mutual dependence between feature and target
- Range: [0, ∞), higher = more dependence
- Advantages: Captures non-linear relationships

**SHAP (SHapley Additive exPlanations)**:
- Purpose: Explain individual predictions
- Output: Feature contribution to prediction
- Advantages: Model-agnostic, theoretically grounded

### B. Model Evaluation Metrics

**AUC-ROC (Area Under Receiver Operating Characteristic Curve)**:
- Range: [0.5, 1.0], higher = better
- Interpretation: Probability that model ranks random positive higher than random negative
- Good: >0.75, Excellent: >0.85

**Precision**: `TP / (TP + FP)`
- Of predicted approvals, what % were actually approved?
- Important when false positives are costly

**Recall**: `TP / (TP + FN)`
- Of actual approvals, what % did we predict?
- Important when false negatives are costly

**F1-Score**: `2 × (Precision × Recall) / (Precision + Recall)`
- Harmonic mean of precision and recall
- Balances both metrics

**Calibration**:
- Are predicted probabilities accurate?
- E.g., Of customers predicted 30% approval, are ~30% actually approved?
- Use calibration plot and Brier score

### C. Feature Engineering Examples

**Debt-to-Income Ratio**:
```python
df['DTI'] = df['Monthly_Housing_Payment'] / df['Monthly_Gross_Income']
```

**Loan-to-Income Ratio**:
```python
df['LTI'] = df['Loan_Amount'] / (df['Monthly_Gross_Income'] × 12)
```

**FICO Bins**:
```python
df['FICO_Bin'] = pd.cut(
    df['FICO_score'],
    bins=[0, 579, 669, 739, 799, 850],
    labels=['Poor', 'Fair', 'Good', 'Very Good', 'Exceptional']
)
```

**Income Quartiles**:
```python
df['Income_Quartile'] = pd.qcut(df['Monthly_Gross_Income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

**Interaction Terms**:
```python
df['FICO_x_Income'] = df['FICO_score'] × df['Monthly_Gross_Income']
```

### D. Revenue Calculation Example

**Current State**:
```python
# Baseline RPA
total_bounty = df['bounty'].sum()  # e.g., $2,750,000
total_apps = len(df)  # 100,000
baseline_rpa = total_bounty / total_apps  # $27.50
```

**Optimized State**:
```python
# Apply matching algorithm
df['optimal_lender'] = df.apply(match_customer, axis=1)
df['predicted_approval'] = df.apply(
    lambda row: predict_approval(row, row['optimal_lender']),
    axis=1
)
df['expected_bounty'] = df['predicted_approval'] × BOUNTY_PER_LENDER[df['optimal_lender']]

# Optimized RPA
optimized_revenue = df['expected_bounty'].sum()  # e.g., $3,200,000
optimized_rpa = optimized_revenue / total_apps  # $32.00

# Incremental revenue
incremental_revenue = optimized_revenue - total_bounty  # $450,000
lift_percentage = (incremental_revenue / total_bounty) × 100  # 16.4%
```

### E. API Specification (Draft)

**Endpoint**: `POST /api/v1/match-lender`

**Request**:
```json
{
  "user_id": "uuid",
  "reason": "debt_consolidation",
  "loan_amount": 50000,
  "fico_score": 720,
  "employment_status": "full_time",
  "employment_sector": "information_technology",
  "monthly_gross_income": 6000,
  "monthly_housing_payment": 1200,
  "ever_bankrupt_or_foreclose": 0
}
```

**Response**:
```json
{
  "assigned_lender": "C",
  "approval_probability": 0.72,
  "expected_bounty": 180,
  "alternative_lenders": [
    {"lender": "A", "probability": 0.45, "expected_bounty": 112},
    {"lender": "B", "probability": 0.38, "expected_bounty": 95}
  ],
  "confidence": "high",
  "model_version": "v1.2.3"
}
```

---

## Document History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-22 | Claude Code | Initial PRD with analysis framework |

---

## Next Steps

### Immediate Actions (Week 1, Day 1)
1. ✅ Load Excel data and validate schema
2. 🔲 Set up project directory structure
3. 🔲 Create virtual environment and install dependencies
4. 🔲 Begin EDA in Jupyter notebook
5. 🔲 Schedule stakeholder kickoff meeting

### Questions to Resolve with Stakeholders
- Confirm bounty structure (fixed per lender vs. variable)
- Clarify lender capacity constraints and volume targets
- Understand approval timing SLAs
- Define acceptable API latency for production
- Review business rules and compliance requirements

**Ready for analysis phase!** 🚀
