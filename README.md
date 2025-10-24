# Lending Optimization Analysis

Data-driven analysis platform to optimize customer-lender matching and maximize Revenue Per Application (RPA) through intelligent assignment algorithms.

## Overview

This project analyzes 100,000 lending applications across 3 lenders (A, B, C) to answer three key business questions:

1. **Variable Importance**: Which customer features predict loan approval?
2. **Lender Preferences**: What customer types do different lenders approve?
3. **Revenue Optimization**: How much incremental revenue can we generate through optimal matching?

**Key Finding**: Optimal customer-lender matching increases revenue by **$2.94M annually (+111.3%)** with 95% confidence interval [$2.91M, $2.97M].

## Technology Stack

- **Frontend**: Next.js 15 + React 19 + TypeScript + Tailwind CSS
- **Backend**: Python 3.11+ (pandas, scikit-learn, xgboost)
- **Analysis**: Jupyter notebooks for exploratory analysis
- **Dashboard**: Interactive validation & results visualization

## Project Structure

```
rv-case-study/
├── app/                    # Next.js frontend & API routes
│   ├── dashboard/          # Analysis dashboard UI
│   └── api/                # Python execution & validation APIs
├── src/                    # Python analysis modules (4 phases)
│   ├── phase1_eda/         # Exploratory data analysis
│   ├── phase2_feature_importance/  # Feature selection
│   ├── phase3_lender_analysis/     # Lender-specific models
│   └── phase4_revenue_optimization/ # Matching algorithm
├── tests/                  # Validation tests for all phases
├── reports/                # Analysis outputs (tables, figures)
├── data/                   # Raw and processed datasets
├── models/                 # Trained ML models
└── documentation/          # PRD and analysis guides
```

## Getting Started

### Prerequisites

- Node.js 18+ and npm
- Python 3.11+
- 4GB RAM minimum

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rv-case-study
   ```

2. **Install Node.js dependencies**
   ```bash
   npm install
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Place the dataset**
   - Add `Pre-Super_Day_candidate_dataset.xlsx` to `app/` directory

### Running the Application

**Start the development server**:
```bash
npm run dev
```

This starts:
- Next.js frontend at http://localhost:3000
- Python Flask backend at http://127.0.0.1:5328

**Access the dashboard**:
- Open http://localhost:3000/dashboard
- Use sidebar to run analysis phases and validate results

## Analysis Workflow

Each phase follows: **Implement → Run → Validate → Review Results**

1. **Phase 1: EDA** - Data exploration and feature engineering
   - 1.1: Univariate analysis
   - 1.2: Bivariate analysis
   - 1.3: Missing value treatment

2. **Phase 2: Feature Importance** - Variable selection
   - 2.1: Statistical importance testing
   - 2.2: ML feature importance (Random Forest, XGBoost)
   - 2.3: Feature validation

3. **Phase 3: Lender Analysis** - Lender specialization
   - 3.1: Approval rate profiling
   - 3.2: Lender-specific predictive models
   - 3.3: Specialization analysis

4. **Phase 4: Revenue Optimization** - Matching algorithm
   - 4.1: Baseline revenue calculation
   - 4.2: Optimal matching algorithm
   - 4.3: Incremental revenue estimation

## Key Results

### Baseline vs Optimal Matching

| Metric | Baseline (Random) | Optimal (Matched) | Gain |
|--------|-------------------|-------------------|------|
| **Total Revenue** | $2,641,500 | $5,582,149 | **+$2,940,649** |
| **Revenue Per App** | $26.42 | $55.82 | **+$29.41** |
| **Approval Rate** | 10.98% | 23.19% | +12.21 pp |

### Top Customer Segments

- **Small Loans** ($7K-$11K): +$1.48M incremental revenue
- **High Income** (Q4): +$1.39M incremental revenue
- **Good FICO** (670+): +$1.31M incremental revenue

### Lender Specializations

- **Lender A**: High FICO (670+), Income Q3-Q4, Medium-large loans
- **Lender B**: Mid FICO (600-670), Medium income, Small-medium loans
- **Lender C**: Broad FICO (580-700), All incomes, Small loans (highest opportunity)

## Validation

Run validation tests to verify analysis correctness:

```bash
# Validate specific phase
python3 tests/test_phase4_incremental_revenue.py

# All validations accessible via dashboard
```

**Phase 4.3 Validation**: ✅ 100% pass rate (35/35 checks)

## Documentation

- **PRD**: `PRD_LENDING_OPTIMIZATION.md` - Full analysis roadmap
- **Executive Summary**: `EXECUTIVE_SUMMARY.md` - Business findings with mathematical derivations
- **Phase Guides**: `documentation/PHASE{1-4}_GUIDE.md` - Implementation details

## Contributing

1. Follow phase-based structure for new analysis
2. Add validation tests for all outputs
3. Update documentation with findings
4. Use dashboard for execution and validation


---

**Questions?** See `PRD_LENDING_OPTIMIZATION.md` for complete analysis methodology and technical details.
