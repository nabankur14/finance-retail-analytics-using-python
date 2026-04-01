# Finance & Retail Analytics — Credit Default Prediction & Market Risk Analysis

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-28a745?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> **Predicting corporate financial defaults using machine learning and analyzing Indian equity portfolio risk through statistical return and volatility modeling.**

---

## Table of Contents
- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Results](#key-results)
- [Business Impact](#business-impact)
- [Skills](#skills)
- [Key Learnings](#key-learnings)
- [Future Improvements](#future-improvements)
- [Repository Structure](#repository-structure)
- [Author](#-author)

---

## Project Overview

This project is a comprehensive, dual-part financial analytics solution designed to address two critical challenges in modern finance:

- **Part A** — Building a machine learning-powered **Financial Health Assessment Tool** to predict whether a company will default (negative net worth) in the following year, using balance sheet metrics spanning 4,256 companies and 51 financial features.
- **Part B** — Conducting a rigorous **Market Risk Analysis** on a portfolio of five Indian stocks over 8 years (418 weeks) to assess risk-return tradeoffs and support data-driven investment decisions.

Together, these analyses form an end-to-end framework for financial risk intelligence — from credit risk modeling to equity portfolio management.

👉 [Open the notebook to explore full analysis](notebook/finance-retail-analytics-using-python.ipynb)

---

## Business Problem

### Part A — Credit Default Prediction

In the modern financial landscape, businesses and investors urgently need reliable mechanisms to assess the creditworthiness of companies before making investment or lending decisions. Traditional manual assessments are slow, subjective, and prone to error.

A group of venture capitalists commissioned the development of a **Financial Health Assessment Tool** — an automated system powered by machine learning to:

- Identify companies at risk of **defaulting on their financial obligations** (net worth turning negative in the next fiscal year)
- Evaluate **credit risk exposure** through liquidity ratios, debt-to-equity ratios, and profitability metrics
- Enable **proactive risk mitigation** strategies before financial distress occurs

**Stakeholders:** Venture capitalists, credit analysts, institutional investors, risk management teams

**Decision Impact:** Informed go/no-go investment decisions, portfolio risk hedging, early warning systems for financial distress

---

### Part B — Market Risk Analysis

Investors face market risk arising from asset price fluctuations driven by economic events, geopolitical developments, and shifting investor sentiment. Quantifying this risk using historical data is essential for building resilient portfolios.

The objective is to analyze a portfolio of five Indian stocks using statistical risk-return modeling to guide allocation decisions and optimize risk-adjusted returns.

**Stakeholders:** Individual investors, portfolio managers, financial advisors

**Decision Impact:** Portfolio allocation strategy, risk categorization of stocks, stop-loss implementation, rebalancing triggers

---

## Dataset

### Part A — Corporate Financial Dataset

| Attribute | Details |
|---|---|
| Source | Balance sheet financial metrics of Indian companies |
| Records | 4,256 companies |
| Features | 51 columns (all numeric — float64 / int64) |
| Target Variable | Derived: `default` = 1 if Networth Next Year < 0, else 0 |
| Class Distribution | Non-Defaulters: 79% (3,352) · Defaulters: 21% (904) |
| Missing Values | 17,778 missing entries across multiple columns |

**Key Features Include:** Total Assets, Net Worth, Total Income, Profit After Tax (PAT), PBDITA, PBT, Borrowings, Debt-to-Equity Ratio, Current Ratio, Quick Ratio, Shareholders' Funds, Cumulative Retained Profits, TOL/TNW, EPS, Adjusted EPS, and 35+ additional financial indicators.

---

### Part B — Indian Stock Price Dataset

| Attribute | Details |
|---|---|
| Source | Weekly closing prices of five Indian stocks |
| Records | 418 weeks (~8 years) |
| Features | Date + 5 stock columns |
| Stocks Covered | ITC Limited, Bharti Airtel, Tata Motors, DLF Limited, Yes Bank |
| Missing Values | None |
| Data Type | 5 numerical (int64), 1 datetime (object → converted) |

---

## Methodology

### Part A — Credit Default Prediction

#### 1. Data Understanding
The dataset was loaded and inspected for structure, data types, and completeness. The dataset contains 4,256 records across 51 columns — all numeric except the index. The target variable `default` was engineered from `Networth_Next_Year`: companies with negative net worth next year were labeled as defaulters (1), and the rest as non-defaulters (0), yielding a class imbalance of 79:21.

#### 2. Exploratory Data Analysis

**Univariate Analysis:**
Box plots were generated for all 51 numerical features, revealing significant right-skewness across most financial variables including Total Assets, Net Worth, Total Income, and Borrowings. Negative values were observed in profitability ratios (PBT%, PAT%, Cash Profit%), indicating loss-making companies. Histograms of key financial variables confirmed that most companies operate at lower income levels, while a small subset reports extreme values — potential outlier candidates.

**Bivariate Analysis:**
A full correlation heatmap was constructed across all numerical variables, followed by a focused heatmap on key financial variables. Notable correlations identified:
- Net Worth & Total Assets: **0.89** — asset-rich companies maintain stronger net worth
- Total Income & Profit After Tax: **0.78** — higher revenue correlates with higher profitability
- Debt-to-Equity Ratio vs. Default: **+0.42** — higher leverage increases default likelihood
- Net Worth vs. Default: **−0.65** — strong negative indicator of financial instability

#### 3. Data Preprocessing

**Outlier Treatment:** IQR method was applied to each column; outliers were replaced with NaN to preserve row count and avoid data loss.

**Missing Value Analysis:** Columns with more than 30% missing values were identified and dropped: PE_on_BSE (67%), Investments (51%), Other_income (46%), Contingent_liabilities (42%), Deferred_tax_liability (42%), Income_from_financial_services (38%), and Change_in_stock (31%).

**Missing Value Imputation:** Remaining missing values were imputed using **K-Nearest Neighbour (KNN) Imputation** to preserve the statistical relationships between features — a critical decision since dropping rows would have eliminated over 65% of actual defaulters.

**Train-Test Split & Scaling:** Data was split 70:30 (train:test). Features were normalized using `StandardScaler` to prepare for logistic regression modeling.

#### 4. Model Building

Two classification models were built with class-weighting to handle the target imbalance:

- **Logistic Regression** (via `statsmodels` Logit) — Interpretable baseline model
- **Random Forest Classifier** — Ensemble model for capturing non-linear patterns

#### 5. Model Tuning & Improvement

**Multicollinearity Treatment (VIF):** Variance Inflation Factors were computed; features with VIF > 5 were identified and iteratively removed to produce reliable coefficients. Highly collinear features included Total Assets (VIF: ∞), Total Liabilities (VIF: ∞), Sales (92.34), Total Income (89.03), and Total Expenses (52.13).

**ROC Curve & Threshold Optimization:** The ROC curve was plotted to determine the optimal classification threshold for the logistic regression model. The resulting AUC of 0.59 confirmed that logistic regression had weak discriminatory ability — only marginally better than random classification.

**Hyperparameter Tuning (RandomSearchCV):** The Random Forest model was tuned using `RandomizedSearchCV` to find optimal hyperparameters, improving test recall from 5% (baseline) to 30% (tuned) while reducing overfitting.

#### 6. Model Evaluation

Models were evaluated using Accuracy, Recall, Precision, F1-Score, and AUC-ROC on both training and test sets.

---

### Part B — Market Risk Analysis

#### 1. Data Understanding
Weekly stock price data for five Indian companies was loaded covering 418 weeks. The Date column was converted from object type to proper `datetime` format. No missing or duplicate values were found.

#### 2. Statistical Summary
Descriptive statistics were computed for each stock. Bharti Airtel showed the highest average price (₹528.26), while Yes Bank had the lowest median price (₹30) — indicating a significant long-term price decline. Bharti Airtel and Tata Motors exhibited the highest standard deviations, while ITC Limited and Bharti Airtel were comparatively stable.

#### 3. Stock Price Analysis
A time-series line chart was plotted for all five stocks from 2016 to 2024. Key observations: Bharti Airtel and Tata Motors showed strong upward trends post-2020; Yes Bank exhibited a dramatic collapse and sustained low prices; Tata Motors and DLF Limited showed recovery trajectories post-COVID dip.

#### 4. Return and Risk Analysis
Weekly percentage returns were computed for each stock using `.pct_change()`. Mean returns and standard deviations were calculated and tabulated:
- **Highest mean return:** DLF Limited (0.004863), Bharti Airtel (0.003271)
- **Negative mean return:** Yes Bank (−0.004737)
- **Highest volatility:** Yes Bank (std: 0.093879)
- **Most stable:** ITC Limited (std: 0.035904)

A **Net Return vs. Volatility scatter plot** was constructed to visually map each stock's risk-return position, clearly separating defensive stocks from aggressive and underperforming ones.

---

## Key Results

### Part A — Model Performance Summary

| Model | Accuracy | Recall | Precision | F1 |
|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.79 | 0.01 | 0.60 | 0.02 |
| Tuned Logistic Regression | 0.79 | 0.01 | 0.67 | 0.01 |
| Random Forest (Baseline) | 0.67 | 0.05 | 0.08 | 0.06 |
| **Tuned Random Forest ✅** | **0.72** | **0.30** | **0.32** | **0.31** |

**Final Model Selected: Tuned Random Forest** — Best balance of recall (30%) and precision (32%) on the test set with minimal overfitting.

**Key Insight:** The dataset's class imbalance (79:21) made recall the critical performance metric — missing an actual defaulter is far costlier than a false alarm in financial risk contexts.

---

### Part B — Risk-Return Summary

| Stock | Mean Weekly Return | Std Deviation | Investor Profile |
|---|---|---|---|
| ITC Limited | 0.001634 | 0.035904 | Risk-Averse |
| Bharti Airtel | 0.003271 | 0.038728 | Risk-Averse / Moderate |
| Tata Motors | 0.002234 | 0.060484 | Growth-Oriented |
| DLF Limited | 0.004863 | 0.057785 | Growth-Oriented |
| Yes Bank | −0.004737 | 0.093879 | High-Risk / Speculative |

---

## Business Impact

### Part A — Credit Risk

1. **Automate Credit Screening:** Deploy the Tuned Random Forest model as an automated early-warning system to flag high-risk companies before lending or investment decisions are made.
2. **Prioritize High-Risk Features:** Focus credit review processes on companies with high debt-to-equity ratios, negative PAT, and declining net worth — the strongest predictors of default.
3. **Dynamic Monitoring:** Implement quarterly re-scoring of portfolio companies using updated financial statements to track drift toward distress.
4. **Capital Allocation Optimization:** Use default probability scores to weight portfolio positions — reducing exposure to companies with high predicted default likelihood.
5. **Debt Restructuring Triggers:** Use model outputs to proactively initiate debt renegotiation conversations with at-risk borrowers before default occurs.

### Part B — Portfolio Risk

1. **Stable Core Holdings:** Allocate the majority of conservative portfolios to ITC Limited and Bharti Airtel — lower volatility, consistent positive returns.
2. **Growth Allocation:** Include DLF Limited and Tata Motors for return-seeking investors with appropriate risk tolerance and a long investment horizon.
3. **Avoid or Limit Yes Bank Exposure:** Negative mean return and extreme volatility make Yes Bank unsuitable for risk-averse strategies; treat only as a speculative, minimal position.
4. **Stop-Loss Implementation:** Apply stop-loss rules for high-volatility holdings (Yes Bank, Tata Motors) to cap downside risk.
5. **Periodic Rebalancing:** Reassess and rebalance portfolio allocations quarterly using updated return and volatility metrics.

---

## Skills

### Technical Skills

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Logistic Regression](https://img.shields.io/badge/Logistic_Regression-8B0000?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Random Forest](https://img.shields.io/badge/Random_Forest-228B22?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Hyperparameter Tuning](https://img.shields.io/badge/Hyperparameter_Tuning-9400D3?style=for-the-badge&logo=scikit-learn&logoColor=white)
![KNN Imputation](https://img.shields.io/badge/KNN_Imputation-FF6347?style=for-the-badge&logo=scikit-learn&logoColor=white)
![VIF/Multicollinearity Analysis](https://img.shields.io/badge/VIF%2FMulticollinearity_Analysis-4682B4?style=for-the-badge&logo=statsmodels&logoColor=white)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-DC143C?style=for-the-badge&logo=scikit-learn&logoColor=white)
![EDA](https://img.shields.io/badge/EDA-FFA500?style=for-the-badge&logo=google-analytics&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-000000?style=for-the-badge&logo=matplotlib&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-77AC1D?style=for-the-badge&logo=seaborn&logoColor=white)
![Statistical Analysis](https://img.shields.io/badge/Statistical_Analysis-4B8BBE?style=for-the-badge&logo=scipy&logoColor=white)
![Credit Risk Modeling](https://img.shields.io/badge/Credit_Risk_Modeling-B22222?style=for-the-badge&logo=bank&logoColor=white)
![Portfolio Risk Analysis](https://img.shields.io/badge/Portfolio_Risk_Analysis-006400?style=for-the-badge&logo=chartdotjs&logoColor=white)
![Feature Engineering](https://img.shields.io/badge/Feature_Engineering-FF8C00?style=for-the-badge&logo=databricks&logoColor=white)
![Class Imbalance Handling](https://img.shields.io/badge/Class_Imbalance_Handling-8A2BE2?style=for-the-badge&logo=scikit-learn&logoColor=white)

### Soft Skills

![Financial Acumen](https://img.shields.io/badge/Financial_Acumen-1E3A5F?style=for-the-badge&logo=bank&logoColor=white)
![Risk Assessment](https://img.shields.io/badge/Risk_Assessment-DC143C?style=for-the-badge&logo=chartdotjs&logoColor=white)
![Analytical Thinking](https://img.shields.io/badge/Analytical_Thinking-4B0082?style=for-the-badge&logo=mindmap&logoColor=white)
![Problem Solving](https://img.shields.io/badge/Problem_Solving-FF4500?style=for-the-badge&logo=brainly&logoColor=white)
![Data Storytelling](https://img.shields.io/badge/Data_Storytelling-2E8B57?style=for-the-badge&logo=google-analytics&logoColor=white)
![Attention to Detail](https://img.shields.io/badge/Attention_to_Detail-00CED1?style=for-the-badge&logo=google-search-console&logoColor=white)
![Business Communication](https://img.shields.io/badge/Business_Communication-25D366?style=for-the-badge&logo=google-messages&logoColor=white)
![Strategic Decision Making](https://img.shields.io/badge/Strategic_Decision_Making-FF8C00?style=for-the-badge&logo=target&logoColor=white)

---

## Key Learnings

- **Class imbalance is a silent model killer** — in financial default prediction, accuracy alone is deeply misleading; recall must be the primary optimization target to avoid missing actual defaulters
- **KNN Imputation over row-dropping** — naive row deletion would have eliminated 65%+ of defaulters; KNN imputation preserved data integrity while handling missing values responsibly
- **VIF-based feature selection** matters beyond just p-values — multicollinearity inflates p-values and masks the true significance of predictors in logistic regression; iterative VIF removal was essential for reliable inference
- **Hyperparameter tuning reduces overfitting** — the baseline Random Forest had a 56-point recall gap between train (61%) and test (5%); tuning reduced this to a 17-point gap, dramatically improving generalization
- **Risk and return must always be evaluated together** — a stock's return in isolation is meaningless; the Sharpe-like analysis combining mean returns and standard deviation revealed that Yes Bank's negative returns with extreme volatility made it categorically different from all other portfolio assets

---

## Future Improvements

1. **Address Class Imbalance with SMOTE / ADASYN** — Apply oversampling techniques on the minority class (defaulters) during training to further improve recall without sacrificing precision
2. **Ensemble Stacking** — Combine Logistic Regression, Random Forest, and Gradient Boosting (XGBoost/LightGBM) in a stacked ensemble to capture both linear and non-linear patterns for improved default prediction
3. **Time-Series Credit Scoring** — Incorporate multi-year financial data per company to model trend-based deterioration rather than a single-year snapshot
4. **Value at Risk (VaR) Modeling** — Extend the market risk analysis with Monte Carlo simulation and Historical VaR to provide confidence-interval-based loss estimates for the equity portfolio
5. **Interactive Dashboard Deployment** — Build a Streamlit or Dash web application to allow investors and analysts to dynamically input financial metrics and receive real-time default probability scores and portfolio risk assessments

---

## Repository Structure

```
finance-retail-analytics-using-python/
│
├── data/
│   ├── Comp_Fin_Data.csv                             # Corporate financial dataset (Part A)
│   └── Market_Risk_Data.csv                          # Stock price dataset (Part B)
│
├── notebook/
│   └── finance-retail-analytics-using-python.ipynb   # Analysis notebook
│
├── requirements.txt                                  # Project dependencies
├── README.md                                         # Project documentation
├── LICENSE                                           # License file
└── .gitignore                                        # Git ignore file
```

---

## 👤 Author

**Nabankur Ray**

Passionate about real-world data-driven solutions

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat&logo=github)](https://github.com/nabankur14) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/nabankur-ray-876582181/)

![GitHub Stats](https://github-readme-stats-eight-theta.vercel.app/api?username=nabankur14&show_icons=true)

---

⭐ If you like this project — give it a ⭐ on GitHub — it helps a lot!