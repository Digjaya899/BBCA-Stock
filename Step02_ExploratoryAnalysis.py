#
# Step02, 2025/12/03
# File: Step02_ExploratoryAnalysis.py
# Short description of the task
#
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy import stats 

# 2.1 Factor Analysis
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

exp_corr = df.select_dtypes(include=["number"]).corr().abs()

mask = np.triu(np.ones_like(exp_corr, dtype=bool), k=1)

pairwise_stats = pd.Series(exp_corr.where(mask).stack()).describe()

print(pairwise_stats)

# 2.2.1 Check Assumptions
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

exp_matrix = df[
    ["Year","Stock Price (Rp)","EPS (Rp)","Dividends (Rp)",
     "P/E","ROA (%)","ROE (%)","Debt-to-Equity",
     "Total Assets (Rp)","Total Liabilities (Rp)","Total Debt (Rp)","Total Equity (Rp)",
     "Revenue (Rp)","Net Profit (Rp)","Operating Cash Flow (Rp)",]
]

corr = exp_matrix.corr()
determinant = np.linalg.det(corr)
eigenvalues = np.linalg.eigvalsh(corr)

assumptions = pd.Series(
    {
        "determinant": round(determinant, 4),
        "min_eigenvalue": round(eigenvalues.min(), 4),
        "max_eigenvalue": round(eigenvalues.max(), 4),
    }
)

print(assumptions)

# 2.2.2 Number of Factors
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

exp_corr = df[
    ["Year","Stock Price (Rp)","EPS (Rp)","Dividends (Rp)",
     "P/E","ROA (%)","ROE (%)","Debt-to-Equity",
     "Total Assets (Rp)","Total Liabilities (Rp)","Total Debt (Rp)","Total Equity (Rp)",
     "Revenue (Rp)","Net Profit (Rp)","Operating Cash Flow (Rp)",]
].corr() 

eigenvalues = np.linalg.eigvalsh(exp_corr)

factor_table = pd.DataFrame(
    {
        "eigenvalue": sorted(eigenvalues, reverse=True),
        "cumulative_variance": np.cumsum(sorted(eigenvalues, reverse=True)) / eigenvalues.sum(),
    }
)
print(factor_table.head())

# 3. Output