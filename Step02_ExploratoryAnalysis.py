#
# Step02, 2025/12/05
# File: Step02_ExploratoryAnalysis.py
# Short description of the task
#
import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# 2.01 DATA PRELOADING
print("2.01 DATA PRELOADING")

if "df" not in globals():
    df = pd.read_csv("BBCA_analyzed.csv")

exp_cols = [col for col in df.columns if col.startswith("analyzedvar_")]
scaler = StandardScaler()
X_std = scaler.fit_transform(df[exp_cols])

exp_corr = df.filter(like="analyzedvar_").corr().abs()
mask = np.triu(np.ones_like(exp_corr, dtype=bool), k=1)
pairwise_stats = pd.Series(exp_corr.where(mask).stack()).describe()

print()
print(pairwise_stats)


# 2.02 SELECTED VARIABLE MEAN STD
print("\n\n2.02 SELECTED VARIABLE MEAN STD")

if "df" not in globals():
    df = pd.read_csv("BBCA_analyzed.csv")

selected= df.filter(like="analyzedvar_").agg(["mean", "std"]).T.round(2).head()

print()
print(selected)


# 2.03 CHECK ASSUMPTIONS (KMO)
print("\n\n2.03 KMO")

if "df" not in globals():
    df = pd.read_csv("BBCA_analyzed.csv")

exp_matrix = df.filter(like="analyzedvar_")
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

print()
print(assumptions)


# 2.04 NUMBER OF FACTORS (EIGENVALUE)
print("\n\n2.04 NUMBER OF FACTORS")

if "df" not in globals():
    df = pd.read_csv("BBCA_analyzed.csv")

exp_corr = df.filter(like="analyzedvar_").corr()
eigenvalues = np.linalg.eigvalsh(exp_corr)
factor_table = pd.DataFrame(
    {
        "eigenvalue": sorted(eigenvalues, reverse=True),
        "cumulative_variance": np.cumsum(sorted(eigenvalues, reverse=True))
        / eigenvalues.sum(),
    }
)

print()
print(factor_table.head())


# 2.05 FACTOR ANALYSIS 
print("\n\n2.05 FACTOR ANALYSIS")

if "df" not in globals():
    df = pd.read_csv("BBCA_analyzed.csv")

exp_cols = [col for col in df.columns if col.startswith("analyzedvar_")]
fa_model = FactorAnalysis(n_components=3, random_state=0)
fa_model.fit(X_std)

factor_loadings = pd.DataFrame(
    fa_model.components_.T,
    index=exp_cols,
    columns=["factor1", "factor2", "factor3"],
)

print()
print(factor_loadings.round(3).head())


# 2.06 FACTOR SCORE
print("\n\n2.06 FACTOR SCORE")

if "df" not in globals():
    df = pd.read_csv("BBCA_analyzed.csv")

exp_cols = [col for col in df.columns if col.startswith("analyzedvar_")]
if "fa_model" not in globals():
    fa_model = FactorAnalysis(n_components=3, random_state=0).fit(X_std)

factor_scores_df = pd.DataFrame(
    fa_model.transform(X_std),
    columns=["factor1_score", "factor2_score", "factor3_score"],
    index=df["Year"],
)

print()
print(factor_scores_df.head().round(3))


# 2.07 VALIDATION
print("\n\n2.07 VALIDATION")

if "df" not in globals():
    df = pd.read_csv("BBCA_analyzed.csv")

exp_cols = [col for col in df.columns if col.startswith("analyzedvar_")]
if "fa_model" not in globals():
    from sklearn.decomposition import FactorAnalysis

    fa_model = FactorAnalysis(n_components=3, random_state=0).fit(df[exp_cols])

loadings = pd.DataFrame(
    fa_model.components_.T,
    index=exp_cols,
    columns=["factor1", "factor2", "factor3"],
)
communalities = (loadings**2).sum(axis=1)
validation_table = pd.DataFrame(
    {
        "communalities": communalities.round(3),
        "uniqueness": (1 - communalities).round(3),
    }
)

print()
print(validation_table.head())


# 2.08 CONVERGENT VALIDITY
print("\n\n2.08COVERGENT VALIDITY")

if "df" not in globals():
    df = pd.read_csv("BBCA_analyzed.csv")

exp_cols = [col for col in df.columns if col.startswith("analyzedvar_")]
if "fa_model" not in globals():
    from sklearn.decomposition import FactorAnalysis

    fa_model = FactorAnalysis(n_components=3, random_state=0).fit(df[exp_cols])

avg_abs_loadings = pd.DataFrame(
    fa_model.components_.T,
    index=exp_cols,
    columns=["factor1", "factor2", "factor3"],
).abs().mean()

print()
print(avg_abs_loadings.to_frame(name="avg_abs_loading").round(3))


# 2.09 DISCRIMINANT VALIDITY
print("\n\n2.09DISCRIMINANT VALIDITY")

if "df" not in globals():
    df = pd.read_csv("BBCA_analyzed.csv")

exp_cols = [col for col in df.columns if col.startswith("analyzedvar_")]
if "fa_model" not in globals():
    from sklearn.decomposition import FactorAnalysis

    fa_model = FactorAnalysis(n_components=3, random_state=0).fit(df[exp_cols])

if "factor_scores_df" not in globals():
    factor_scores_df = pd.DataFrame(
        fa_model.transform(df[exp_cols]),
        columns=["factor1_score", "factor2_score", "factor3_score"],
        index=df["Year"],
    )

print()
print(factor_scores_df.corr().round(3))


# 2.10 VALIDITY SUMMARY REPORT
print("\n\n2.10VALIDITY SUMMARY REPORT")

import pandas as pd
from sklearn.decomposition import FactorAnalysis

if "df" not in globals():
    df = pd.read_csv("BBCA_analyzed.csv")

exp_cols = [col for col in df.columns if col.startswith("analyzedvar_")]
if "fa_model" not in globals():
    fa_model = FactorAnalysis(n_components=3, random_state=0).fit(df[exp_cols])

loadings = pd.DataFrame(
    fa_model.components_.T,
    index=exp_cols,
    columns=["factor1", "factor2", "factor3"],
)

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

summary = pd.DataFrame(summary_rows)

print()
print(summary)