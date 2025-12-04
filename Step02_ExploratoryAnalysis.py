#
# Step02, 2025/12/05
# File: Step02_ExploratoryAnalysis.py
# Short description of the task
#
import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis

# 2.1 Check Assumptions (KMO)
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

exp_matrix = df[["Stock_Price_Rp","Dividends_Rp", "EPS_Rp","PE","ROA_pct","ROE_pct","Debt_to_Equity",]]
                 
corr = exp_matrix.corr()
determinant = np.linalg.det(corr)
eigenvalues = np.linalg.eigvalsh(corr)

KMO= pd.Series(
    {
        "determinant": round(determinant, 4),
        "min_eigenvalue": round(eigenvalues.min(), 4),
        "max_eigenvalue": round(eigenvalues.max(), 4),
    }
)
print(KMO)

# 2.2 Number of Factors (Eigenvalue)
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

exp_corr = df[["Stock_Price_Rp","Dividends_Rp", "EPS_Rp","PE","ROA_pct","ROE_pct","Debt_to_Equity",]].corr()
eigenvalues = np.linalg.eigvalsh(exp_corr)
factor_table = pd.DataFrame(
    {
        "eigenvalue": sorted(eigenvalues, reverse=True),
        "cumulative_variance": np.cumsum(sorted(eigenvalues, reverse=True))
        / eigenvalues.sum(),
    }
)
print(factor_table)

# 2.3 Factor Analysis 


# 2. Process

# 3. Output