#
# Test3, 2025/12/05
# File: Test3.py
# Short description of the task
#
import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Input

# 2. Process

# 3. Output

if "df" not in globals():
    df = pd.read_csv("BBCA_analyzed.csv")

exp_cols = [col for col in df.columns if col.startswith("analyzedvar_")]

print("\n\n# 3.04 STATISTICAL SIGNIFICANCE TESTING")

if "factor_scores_df" not in globals():
    scaler = StandardScaler()
    X_std = scaler.fit_transform(df[exp_cols])
    fa_model = FactorAnalysis(n_components=3, random_state=0).fit(X_std)
    factor_scores_df = pd.DataFrame(
        fa_model.transform(X_std),
        columns=["factor1_score", "factor2_score", "factor3_score"],
        index=df["Year"],
    )

merged = pd.concat([df, factor_scores_df], axis=1)

tests = []
for feature in ["analyzedvar_PE", "analyzedvar_ROA_pct", "factor1_score"]:
    subset = merged[["analyzedvar_Stock_Price_Rp", feature]].dropna()
    r, p = stats.pearsonr(subset["analyzedvar_Stock_Price_Rp"], subset[feature])
    tests.append(
        {"feature": feature, "r": round(r, 3), "p_value": round(p, 3), "n": len(subset)}
    )

print()
print(pd.DataFrame(tests))