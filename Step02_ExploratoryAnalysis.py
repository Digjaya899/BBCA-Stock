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

# 2.1.1 Factor Analysis
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

corr = df[["Stock_Price_Rp","EPS_Rp","Dividends_Rp",
           "PE","ROA_pct","ROE_pct","Debt_to_Equity",]].corr().abs()

mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

pairwise_stats = pd.Series(corr.where(mask).stack()).describe()

print(pairwise_stats)

# 2.1.2 Process
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

df[["Stock_Price_Rp","EPS_Rp","Dividends_Rp",
           "PE","ROA_pct","ROE_pct","Debt_to_Equity",
    ]].agg(["mean", "std"]).T.round(2).head()


# 2.2.1 Check Assumptions
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

exp_matrix = df[
    ["Stock_Price_Rp","EPS_Rp","Dividends_Rp",
     "PE","ROA_pct","ROE_pct","Debt_to_Equity",]]

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
    ["Stock_Price_Rp","EPS_Rp","Dividends_Rp",
     "PE","ROA_pct","ROE_pct","Debt_to_Equity",]].corr() 

eigenvalues = np.linalg.eigvalsh(exp_corr)

factor_table = pd.DataFrame(
    {"eigenvalue": sorted(eigenvalues, reverse=True),
     "cumulative_variance": np.cumsum(sorted(eigenvalues, reverse=True)) / eigenvalues.sum(),})

print(factor_table.head())

# 2.2.3 Factor Analysis
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

exp_cols = ["Stock_Price_Rp","EPS_Rp","Dividends_Rp",
            "PE","ROA_pct","ROE_pct","Debt_to_Equity",]

fa_model = FactorAnalysis(n_components=3, random_state=0)
fa_model.fit(df[exp_cols])

factor_loadings = pd.DataFrame(
    fa_model.components_.T,
    index=exp_cols,
    columns=["factor1", "factor2", "factor3"],)

print(factor_loadings.round(3).head())

# 2.2.4 Factor Scores
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

exp_cols = ["Stock_Price_Rp","EPS_Rp","Dividends_Rp",
            "PE","ROA_pct","ROE_pct","Debt_to_Equity",]

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

exp_cols = ["Stock_Price_Rp","EPS_Rp","Dividends_Rp",
            "PE","ROA_pct","ROE_pct","Debt_to_Equity",]

if "fa_model" not in globals():
    fa_model = FactorAnalysis(n_components=3, random_state=0).fit(df[exp_cols])

loadings = pd.DataFrame(
    fa_model.components_.T,
    index=exp_cols,
    columns=["factor1", "factor2", "factor3"],)

communalities = (loadings**2).sum(axis=1)
validation_table = pd.DataFrame(
    {"communalities": communalities.round(3),
     "uniqueness": (1 - communalities).round(3),})

print(validation_table)

# 2.4.1 Convergent Validality
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

exp_cols = ["Stock_Price_Rp","EPS_Rp","Dividends_Rp",
            "PE","ROA_pct","ROE_pct","Debt_to_Equity",]

if "fa_model" not in globals():
    fa_model = FactorAnalysis(n_components=3, random_state=0,).fit

avg_abs_loadings = pd.DataFrame(
    fa_model.components_.T,
    index=exp_cols,
    columns=["factor 1", "factor 2", "factor 3"],
).abs().mean()

print(avg_abs_loadings.to_frame(name="avg_abs_loading").round(3))

# 2.4.2 Discriminant Validity 
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

exp_cols = ["Stock_Price_Rp","EPS_Rp","Dividends_Rp",
            "PE","ROA_pct","ROE_pct","Debt_to_Equity",]

if "fa_model" not in globals():
    fa_model = FactorAnalysis(n_components=3, random_state=0).fit(df[exp_cols])

if "factor_scores_df" not in globals():
    factor_scores_df = pd.DataFrame(
        fa_model.transform(df[exp_cols]),
        columns=["factor1_score", "factor2_score", "factor3_score"],
        index=df["Year"],
    )

print(factor_scores_df.corr().round(3))

# 2.5 Summary Report

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

exp_cols = ["Stock_Price_Rp","EPS_Rp","Dividends_Rp",
            "PE","ROA_pct","ROE_pct","Debt_to_Equity",]

if "fa_model" not in globals():
    fa_model = FactorAnalysis(n_components=3, random_state=0).fit(df[exp_cols])

loadings = pd.DataFrame(
    fa_model.components_.T,
    index=exp_cols,
    columns=["factor1", "factor2", "factor3"],)

summary_rows = []
for factor in loadings.columns:
    top = loadings[factor].abs().sort_values(ascending=False).head(3)
    summary_rows.append(
        {
            "factor": factor,
            "top_item_1": top.index[0],
            "loading_1": round(loadings.loc[top.index[0], factor], 3),
            "top_item_2": top.index[1],
            "loading_2": round(loadings.loc[top.index[1], factor], 3),
            "top_item_3": top.index[2],
            "loading_3": round(loadings.loc[top.index[2], factor], 3),
        }
    )

print(pd.DataFrame(summary_rows))