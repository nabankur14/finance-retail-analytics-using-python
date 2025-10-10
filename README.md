# Finance & Retail Analytics Using Python

**Author:** Nabankur Ray 

**Short description:**  
End-to-end Finance & Retail Analytics project with EDA, preprocessing, Logistic Regression and Tuned Random Forest for company default prediction, plus stock return & volatility analysis for five Indian stocks.

---

## Contents
- `/notebooks` — exploratory and modeling Jupyter notebooks  
- `/data` — processed CSVs and sample data (sensitive/raw data excluded)   
- `BUSINESS_REPORT-FRA.pdf` — full business & technical report

---

# 🧠 Finance & Retail Analytics (FRA) — Jupyter Notebooks

This folder contains all the **notebooks** used in the _Finance & Retail Analytics (FRA)_ project, demonstrating a complete end-to-end data science workflow — from data cleaning and feature engineering to model development and business insights.

---

## 📁 Folder Overview

| File Name | Description |
|------------|--------------|
| **FRA_Coded_Project_Finalised.ipynb** | Final integrated notebook containing complete project workflow: data preparation, EDA, feature engineering, model training, evaluation, and business insights. |
| *(optional)* `EDA_Insights.ipynb` | (If you separate later) Focused on exploratory data analysis — summary stats, visualizations, and trend identification. |
| *(optional)* `Model_Tuning.ipynb` | (If you separate later) Hyperparameter tuning for Random Forest and model performance comparison. |
| *(optional)* `Stock_Risk_Analysis.ipynb` | Stock market risk-return and volatility analysis for Indian companies. |

---

## ⚙️ Project Workflow (Inside the Final Notebook)

1. **Importing Libraries & Loading Data**
   - Libraries: `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `statsmodels`
   - Loads company financial and stock market data.

2. **Data Cleaning & Preprocessing**
   - Handles missing values, detects outliers, and performs data type corrections.
   - Encodes categorical variables and scales numerical features.

3. **Exploratory Data Analysis (EDA)**
   - Visualizes key metrics using barplots, histograms, boxplots, and correlation heatmaps.
   - Identifies patterns in profitability, leverage, liquidity, and stock performance.

4. **Feature Engineering**
   - Creates new ratio-based features (e.g., debt-to-equity, net worth ratios).
   - Applies transformations to normalize skewed distributions.

5. **Model Building**
   - **Models Used:** Logistic Regression, Random Forest (baseline & tuned).
   - Implements hyperparameter tuning using `RandomizedSearchCV`.

6. **Model Evaluation**
   - Evaluates models using Accuracy, Recall, Precision, F1 Score, and ROC-AUC.
   - Final model selected: **Tuned Random Forest** for better recall and interpretability.

7. **Business Insights & Recommendations**
   - Identifies key drivers of company default risk.
   - Provides actionable recommendations for investment and credit decisions.
   - Includes visual risk-return analysis of five Indian stocks.


