#
# Step03, 2025/12/05
# File: Step03_CorrelationCausation.py
# Short description of the task
#
import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Input Prior Data
if "df" not in globals():
    df = pd.read_csv("BBCA_analyzed.csv")

exp_cols = [col for col in df.columns if col.startswith("analyzedvar_")]

scaler = StandardScaler()
X_std = scaler.fit_transform(df[exp_cols])

fa_model = FactorAnalysis(n_components=3, random_state=0).fit(X_std)
factor_scores_df = pd.DataFrame(
    fa_model.transform(X_std),
    columns=["factor1_score", "factor2_score", "factor3_score"],
)

df = df.reset_index(drop=True)
factor_scores_df = factor_scores_df.reset_index(drop=True)


# 3.01 DATA PRELOADING
print("# 3.01 DATA PRELOADING")

target_corr = (
    df.corr(numeric_only=True)["analyzedvar_Stock_Price_Rp"]
      .drop("analyzedvar_Stock_Price_Rp").drop("Year")
      .sort_values(ascending=False)
      .head(5)
)

print()
print(target_corr.round(3))


# 3.02 SELECTED VARIABLE MEAN STD
print("\n\n# 3.02 SELECTED VARIABLE MEAN STD")

eda_frame = pd.concat(
    [
        df[
            [
                "analyzedvar_Stock_Price_Rp",
                "analyzedvar_EPS_Rp",
                "analyzedvar_Dividends_Rp",
            ]
        ],
        factor_scores_df,
    ],
    axis=1,
)

print()
print(eda_frame.describe().loc[["mean", "std"]])


# 3.03 CORRELATION MATRIX
print("\n\n# 3.03 CORRELATION MATRIX")

corr_cols = [
    "analyzedvar_Stock_Price_Rp",
    "analyzedvar_EPS_Rp",
    "analyzedvar_Dividends_Rp",
    "analyzedvar_PE",
]

corr_frame = pd.concat([df[corr_cols], factor_scores_df], axis=1).corr()

print()
print(corr_frame.round(3))


# 3.04 STATISTICAL SIGNIFICANCE TESTING
print("\n\n# 3.04 STATISTICAL SIGNIFICANCE TESTING")

merged = pd.concat([df, factor_scores_df], axis=1)

tests = []
for feature in ["analyzedvar_PE", "analyzedvar_ROA_pct", "factor1_score"]:
    cols = ["analyzedvar_Stock_Price_Rp", feature]
    subset = merged[cols].dropna()

    if len(subset) < 2:
        tests.append(
            {"feature": feature, "r": None, "p_value": None, "n": len(subset)}
        )
        continue

    r, p = stats.pearsonr(subset["analyzedvar_Stock_Price_Rp"], subset[feature])
    tests.append(
        {"feature": feature, "r": round(r, 3), "p_value": round(p, 3), "n": len(subset)}
    )

print()
print(pd.DataFrame(tests))


# 3.05 PARTIAL CORRELATIONS
print("\n\n# 3.05 PARTIAL CORRELATIONS")

def residuals(y, X):
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model.resid

data = df[
    [
        "analyzedvar_Stock_Price_Rp",
        "analyzedvar_EPS_Rp",
        "analyzedvar_Dividends_Rp",
        "analyzedvar_ROA_pct",
    ]
].dropna()

res_target = residuals(
    data["analyzedvar_Stock_Price_Rp"],
    data[["analyzedvar_ROA_pct"]],
)
res_eps = residuals(
    data["analyzedvar_EPS_Rp"],
    data[["analyzedvar_ROA_pct"]],
)
res_div = residuals(
    data["analyzedvar_Dividends_Rp"],
    data[["analyzedvar_ROA_pct"]],
)

partial_summary = pd.Series(
    {
        "price_vs_eps|roa": np.corrcoef(res_target, res_eps)[0, 1],
        "price_vs_div|roa": np.corrcoef(res_target, res_div)[0, 1],
    }
).round(3)

print()
print(partial_summary)


# 3.06 MULTICOLLINEARITY CHECKING
print("\n\n# 3.06 MULTICOLLINEARITY CHECKING")

vif_data = pd.concat(
    [
        df[
            [
                "analyzedvar_EPS_Rp",
                "analyzedvar_Dividends_Rp",
                "analyzedvar_PE",
            ]
        ],
        factor_scores_df,
    ],
    axis=1,
).dropna()

if vif_data.shape[0] < 2 or vif_data.shape[1] == 0:
    print(
        "Not enough data to compute VIF (rows:",
        vif_data.shape[0],
        ", columns:",
        vif_data.shape[1],
        ").",
    )
else:
    X_vif = sm.add_constant(vif_data)
    vif_table = pd.DataFrame(
        {
            "feature": X_vif.columns,
            "VIF": [
                variance_inflation_factor(X_vif.values, i)
                for i in range(X_vif.shape[1])
            ],
        }
    )
    
    print()
    print(vif_table.round(2))


# 3.07 ASSUMPTION CHECKING
print("\n\n# 3.07 ASSUMPTION CHECKING")

analysis_df = pd.concat(
    [
        df[
            [
                "analyzedvar_Stock_Price_Rp",
                "analyzedvar_EPS_Rp",
                "analyzedvar_Dividends_Rp",
            ]
        ],
        factor_scores_df[["factor1_score"]],
    ],
    axis=1,
).dropna()

if analysis_df.shape[0] < 2:
    print(
        "Not enough data to run OLS for assumption checking (rows:",
        analysis_df.shape[0],
        ").",
    )
else:
    X_reg = sm.add_constant(
        analysis_df[
            ["analyzedvar_EPS_Rp", "analyzedvar_Dividends_Rp", "factor1_score"]
        ]
    )
    model = sm.OLS(analysis_df["analyzedvar_Stock_Price_Rp"], X_reg).fit()

    shapiro_p = stats.shapiro(model.resid).pvalue
    bp_test = sm.stats.diagnostic.het_breuschpagan(model.resid, X_reg)[3]

    assumption_series = pd.Series(
        {"shapiro_p": round(shapiro_p, 3), "breusch_p": round(bp_test, 3)}
    )
    
    print()
    print(assumption_series)