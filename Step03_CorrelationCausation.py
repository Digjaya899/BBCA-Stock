#
# Step03, 2025/12/04
# File: Step03_CorrelationCausation.py
# Short description of the task
#
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy import stats 
from sklearn.decomposition import FactorAnalysis

# 3.1 Purpose
import pandas as pd

if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

target_corr = (
    df.corr(numeric_only=True)["Stock_Price_Rp"]
      .drop("Stock_Price_Rp").drop("Year")
      .sort_values(ascending=False)
      .head(5)
)

target_corr.round(3)

# 3.2.1 Mean Std Process
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

exp_cols = ["Stock_Price_Rp","EPS_Rp","Dividends_Rp",
            "PE","ROA_pct","ROE_pct","Debt_to_Equity",]

if "factor_scores_df" not in globals():
    fa_model = FactorAnalysis(n_components=3, random_state=0).fit(df[exp_cols])
    factor_scores_df = pd.DataFrame(
        fa_model.transform(df[exp_cols]),
        columns=["factor1_score", "factor2_score", "factor3_score"],
        index=df["Year"],)

eda_frame = pd.concat(
    [df[["Stock_Price_Rp","EPS_Rp","Dividends_Rp",
            "PE","ROA_pct","ROE_pct","Debt_to_Equity",]],
     factor_scores_df,],
    axis=1,)

eda_frame.describe().loc[["mean", "std"]]

# 2. Process

# 3. Output