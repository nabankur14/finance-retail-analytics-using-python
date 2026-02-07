# 📊 Finance Retail Analytics: Credit Risk Assessment

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 🚀 Project Overview

**Problem Statement**: In the modern financial landscape, accurately predicting corporate default risk is critical. Venture capitalists and investors need a robust "Financial Health Assessment Tool" to evaluate whether a company will maintain a positive net worth in the coming year.

**Objective**: Develop a predictive model to identify potential defaulters based on historical balance sheet data, enabling proactive risk mitigation.

**Business Impact**:
-   **Risk Reduction**: Early identification of default-prone companies.
-   **Investment Confidence**: Data-driven insights for smarter capital allocation.
-   **Strategic Planning**: Helps businesses improve debt management practices.

## 📂 Repository Structure

```
finance-retail-analytics-using-python/
│
├── data/
│   ├── raw/             # Original financial datasets
│   └── processed/       # Cleaner data for modeling
│
├── notebooks/
│   └── analysis.ipynb   # Main analysis notebook
│
├── src/                 # Source code for reproducibility
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   └── evaluation.py
│
├── visuals/             # Charts and diagrams
│
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation
```

## 📊 Dataset

The dataset consists of financial metrics from various companies, including:
-   **Net Worth**: Assets minus liabilities.
-   **Profitability Ratios**: PBT, PBDITA, PAT as % of total income.
-   **Liquidity Ratios**: Current ratio, Quick ratio.
-   **Turnover Ratios**: Debtors turnover, Creditors turnover.
-   **Target Variable**: `default` (1 if Net Worth Next Year <= 0, else 0).

## 🛠️ Methodology

1.  **Data Preprocessing**:
    -   Cleaning column names.
    -   Handling outliers using IQR.
    -   Imputing missing values using `KNNImputer`.
    -   Scaling features using `StandardScaler`.
2.  **Feature Engineering**:
    -   Multicollinearity check using **VIF (Variance Inflation Factor)**.
    -   Removing highly correlated features (VIF > 5).
3.  **Modeling**:
    -   **Logistic Regression**: Interpretable baseline model.
    -   **Random Forest**: Ensemble method to capture non-linear relationships, optimized via `RandomizedSearchCV`.
4.  **Evaluation**:
    -   Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
    -   Key Focus: Maximizing **Recall** to minimize false negatives (missed defaulters).

## 📈 Key Results

| Model | Accuracy | Recall | Principal Metric |
|-------|----------|--------|------------------|
| Logistic Regression | ~90% | ~85% | Good baseline |
| **Random Forest** | **~96%** | **~92%** | **Best Performance** |

*Note: Results may vary slightly depending on random seeds.*

## 💡 Business Recommendations

1.  **Focus on Liquidity**: Companies with low current ratios are significantly more likely to default.
2.  **Debt Management**: High debt-to-equity ratios are a primary red flag.
3.  **Profitability**: Consistent negative PBT is a strong predictor of future default.
4.  **Early Warning System**: Deploy the Random Forest model to flag high-risk portfolios for manual review.

## 💻 How to Run

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/nabankur14/finance-retail-analytics-using-python.git
    cd finance-retail-analytics-using-python
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the analysis**:
    Open `notebooks/analysis.ipynb` in Jupyter Lab or Notebook.

## 👤 Author

**Nabankur Ray**
*Data Scientist | Business Analyst | ML Engineer*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/nabankur14)

---
*Created as part of a Data Science Portfolio Project.*
