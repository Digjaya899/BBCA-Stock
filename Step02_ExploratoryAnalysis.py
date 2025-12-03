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
from sklearn.decomposition import FactorAnalysis

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
    ["ROA (%)","ROE (%)","Debt-to-Equity", "P/E",
     "Stock Price (Rp)","EPS (Rp)","Dividends (Rp)",]
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
    ["ROA (%)","ROE (%)","Debt-to-Equity", "P/E",
     "Stock Price (Rp)","EPS (Rp)","Dividends (Rp)",]
].corr() 

eigenvalues = np.linalg.eigvalsh(exp_corr)

factor_table = pd.DataFrame(
    {
        "eigenvalue": sorted(eigenvalues, reverse=True),
        "cumulative_variance": np.cumsum(sorted(eigenvalues, reverse=True)) / eigenvalues.sum(),
    }
)
print(factor_table.head())

# 2.2.3 Factor Analysis
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

exp_cols = ["ROA (%)","ROE (%)","Debt-to-Equity", "P/E",
            "Stock Price (Rp)","EPS (Rp)","Dividends (Rp)",]

fa_model = FactorAnalysis(n_components=3, random_state=0)
fa_model.fit(df[exp_cols])

factor_loadings = pd.DataFrame(
    fa_model.components_.T,
    index=exp_cols,
    columns=["factor1", "factor2", "factor3"],
)

print(factor_loadings.round(3).head())

# 2.2.4 Factor Scores
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

exp_cols = ["ROA (%)","ROE (%)","Debt-to-Equity", "P/E",
            "Stock Price (Rp)","EPS (Rp)","Dividends (Rp)",]

if "fa_model" not in globals():
    fa_model = FactorAnalysis(n_components=3, random_state=0).fit(df[exp_cols])

factor_scores_df = pd.DataFrame(
    fa_model.transform(df[exp_cols]),
    columns=["factor1_score", "factor2_score", "factor3_score"],
    index=df["Year"],
)

print(factor_scores_df.round(3).head())

# 2.3 Validation
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

exp_cols = ["ROA (%)","ROE (%)","Debt-to-Equity", "P/E",
            "Stock Price (Rp)","EPS (Rp)","Dividends (Rp)",]

if "fa_model" not in globals():
    fa_model = FactorAnalysis(n_components=3, random_state=0).fit(df[exp_cols])

loadings = pd.DataFrame(
    fa_model.components_.T,
    index=exp_cols,
    columns=["factor1", "factor2", "factor3"],)

communalities = (loadings**2).sum(axis=1)
validation_table = pd.DataFrame(
    {
        "communalities": communalities.round(3),
        "uniqueness": (1 - communalities).round(3),
    })

print(validation_table)
