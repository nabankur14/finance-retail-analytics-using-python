<h1 align="center" style="color:#2b7a78;">🧠 Finance & Retail Analytics (FRA) Project</h1>
<h3 align="center" style="color:#17252a;">Predictive Financial Health Assessment & Market Risk Analysis</h3>

<p align="center">
  <strong>Author:</strong> <a href="https://github.com/NabankurRay" target="_blank" style="color:#3aafa9;">Nabankur Ray</a>  
</p>

<hr>

<h2 style="color:#17252a;">📘 Overview</h2>
<p>
This project focuses on <strong>Finance & Retail Analytics (FRA)</strong> — developing a <strong>Financial Health Assessment Tool</strong> that predicts company defaults 
and performs <strong>Market Risk Analysis</strong> on Indian stock portfolios.  
By combining <em>machine learning</em> with <em>financial analytics</em>, it enables businesses and investors to make smarter, data-driven decisions.
</p>

<details open>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">🎯 Objective</summary>
  <p>
  The goal is to:
  <ul>
    <li>Predict whether a company will be a <strong>financial defaulter</strong> based on balance sheet data.</li>
    <li>Analyze <strong>market risk and returns</strong> across key Indian stocks for portfolio optimization.</li>
    <li>Provide <strong>business insights</strong> to strengthen debt management and risk strategies.</li>
  </ul>
  </p>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">📂 Dataset</summary>
  <ul>
    <li><strong>Source:</strong> Simulated corporate financial data & publicly available stock market data.</li>
    <li><strong>Size:</strong> 4,256 companies × 51 financial variables; 418 weeks × 5 stock prices.</li>
    <li><strong>Key Features:</strong>
      <ul>
        <li>Net Worth, Total Assets, Borrowings, Profit After Tax, Total Liabilities</li>
        <li>Debt-to-Equity Ratio, Quick Ratio, Cash Profit %, Contingent Liabilities</li>
        <li>Stock Prices: ITC, Bharti Airtel, Tata Motors, DLF, Yes Bank</li>
      </ul>
    </li>
  </ul>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">🔍 Methodology</summary>
  <ol>
    <li><strong>Data Cleaning & Preprocessing:</strong> Outlier detection via IQR, KNN imputation, feature scaling.</li>
    <li><strong>Exploratory Data Analysis (EDA):</strong> Distribution, correlation heatmaps, and financial ratio insights.</li>
    <li><strong>Feature Engineering:</strong> Derived “Defaulter” target variable, handled multicollinearity using VIF.</li>
    <li><strong>Model Development:</strong> Logistic Regression and Random Forest for prediction.</li>
    <li><strong>Model Evaluation:</strong> Metrics – Accuracy, Precision, Recall, F1-score, and AUC-ROC.</li>
    <li><strong>Insights & Recommendations:</strong> Actionable business and investment strategies.</li>
  </ol>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">🧮 Tools & Technologies</summary>
  <p>
  <code>Python</code>, <code>Pandas</code>, <code>NumPy</code>, <code>Matplotlib</code>, <code>Seaborn</code>,  
  <code>Scikit-learn</code>, <code>Statsmodels</code>, <code>SQL</code>, <code>Power BI</code>, <code>Jupyter Notebook</code>
  </p>
</details>

<details open>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">📊 Results & Insights</summary>
  <p>
  <strong>Model Outcome:</strong> The <strong>Tuned Random Forest</strong> model performed best with:
  <ul>
    <li>Accuracy: <strong>72%</strong></li>
    <li>Recall (Defaulters): <strong>30%</strong></li>
    <li>Precision: <strong>32%</strong></li>
    <li>AUC-ROC: <strong>0.92</strong></li>
  </ul>
  <p>
  The model achieved a solid balance between accuracy and recall, making it effective for early financial distress detection.  
  In stock analysis, <em>DLF Limited</em> and <em>Bharti Airtel</em> demonstrated strong return-to-risk ratios, while <em>Yes Bank</em> showed high volatility and negative returns.  
  Visuals are stored in the <code>/visuals</code> directory.
  </p>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">🚀 Future Scope</summary>
  <ul>
    <li>Integrate <strong>deep learning</strong> for improved predictive performance.</li>
    <li>Incorporate <strong>macroeconomic indicators</strong> and sentiment data.</li>
    <li>Deploy a <strong>web-based dashboard</strong> for real-time analytics.</li>
    <li>Enable <strong>automated portfolio alerts</strong> and investment recommendations.</li>
  </ul>
</details>

<details>
  <summary style="cursor:pointer; color:#3aafa9; font-weight:bold;">📁 Folder Structure</summary>
  <pre style="background:#f0f0f0; padding:10px; border-radius:8px;">
/notebooks     → Jupyter notebooks for analysis and modeling
/data          → Processed and sample datasets
/visuals       → Graphs and EDA visualizations
BUSINESS_REPORT-FRA.pdf → Full business & technical report
README.md      → Project documentation (this file)
  </pre>
</details>

<hr>
<p align="center" style="font-size:14px; color:#555;">
© 2025 <strong>Nabankur Ray</strong> | Data Science & Business Analytics Portfolio  
</p>
