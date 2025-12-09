#
# BBCA Full Analysis
# James Digjaya (G14460702)
# CSV: BBCA_analyzed.csv
#

# 1. Input

# 2. Process

# 3. Output

import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# 1.01 DATA PRELOADING
print("1.01 DATA PRELOADING")

if "df" not in globals():
    df = pd.read_csv("BBCA_analyzed.csv")

summary = df[
    ["analyzedvar_Stock_Price_Rp", "analyzedvar_EPS_Rp", "analyzedvar_Dividends_Rp"]
].describe().loc[["count", "mean", "std", "min", "max"]]

print()
print(summary)


# 1.02 DATA VALIDITY CHECKS
print("\n\n1.02 DATA VALIDITY CHECKS")

likert_columns = [
    "analyzedvar_PE",
    "analyzedvar_ROA_pct",
    "analyzedvar_ROE_pct",
] + [col for col in df.columns if col.startswith("analyzedvar_")]

likert_out_of_range = (
    (df[likert_columns] < 0) | (df[likert_columns] > 100)
).sum().sum()

numeric_checks = {
    "negative_assets": int((df["Total_Assets_BnRp"] < 0).sum()),
    "negative_revenue": int((df["Revenue_BnRp"] < 0).sum()),
}

validity_df = pd.DataFrame(
    {
        "metric": ["likert_out_of_range"] + list(numeric_checks.keys()),
        "violations": [int(likert_out_of_range)] + list(numeric_checks.values()),
    }
)

print()
print(validity_df)


# 1.03 MISSING DATA ANALYSIS
print("\n\n1.03 MISSING DATA ANALYSIS")

missing_counts = df.isna().sum()

print()
print(missing_counts[missing_counts > 0])


# 1.04 OUTLIER DETECTION
print("\n\n1.04 OUTLIER DETECTION")

z_scores = np.abs(
    stats.zscore(
        df[["analyzedvar_Stock_Price_Rp", "analyzedvar_EPS_Rp"]].dropna()
    )
)
outlier_mask = z_scores > 3

outliers = df.loc[
    outlier_mask.any(axis=1),
    ["Year", "analyzedvar_Stock_Price_Rp", "analyzedvar_EPS_Rp"],
]

print()
print(outliers)


# 1.05 RELIABILITY TESTING (CRONBACH'S ALPHA)
print("\n\n1.05 RELIABILITY TESTING (CRONBACH'S ALPHA)")

service_items = df[
    [
        "analyzedvar_EPS_Rp",
        "analyzedvar_Dividends_Rp",
        "analyzedvar_PE",
        "analyzedvar_ROE_pct",
    ]
]
item_var = service_items.var(axis=0, ddof=1)
total_var = service_items.sum(axis=1).var(ddof=1)

alpha = len(service_items.columns) / (len(service_items.columns) - 1) * (
    1 - item_var.sum() / total_var
)

print()
print(round(alpha, 3))


# 1.06 VALIDATION CHECKLIST
print("\n\n1.06 VALIDATION CHECKLIST")

validation_series = pd.Series(
    {
        "rows": df.shape[0],
        "features": df.shape[1],
        "numeric_columns": int(df.select_dtypes(include=["number"]).shape[1]),
        "rows_with_missing": int(df.isna().any(axis=1).sum()),
        "int_columns": int((df.dtypes == "int64").sum()),
        "float_columns": int((df.dtypes == "float64").sum()),
    }
)

print()
print(validation_series)


# 1.07 SUMMARY
print("\n\n1.07 SUMMARY")

vars_for_table = [
    "analyzedvar_Stock_Price_Rp",   # DV
    "analyzedvar_Dividends_Rp",     # IV
    "analyzedvar_EPS_Rp",           # IV
    "analyzedvar_PE",               # IV
    "analyzedvar_ROA_pct",          # IV
    "analyzedvar_ROE_pct",          # IV
    "analyzedvar_Debt_to_Equity",   # IV
    "analyzedvar_EBITDA_BnRp",      # IV
    "Year",                         # Control
]

sub = df[vars_for_table]

desc = sub.describe().T
missing = sub.isna().sum()
z = (sub - sub.mean()) / sub.std(ddof=0)
outliers = (np.abs(z) > 3).sum()

summary_table = pd.DataFrame({
    "Variable": desc.index,
    "N": desc["count"].astype(int),
    "Mean": desc["mean"],
    "Std_Dev": desc["std"],
    "Min": desc["min"],
    "Max": desc["max"],
    "Missing": missing.values,
    "Outliers": outliers.values,
})

print()
print((summary_table).round(2))
summary_table.to_csv("BBCA_summary_stats_all_vars.csv", index=False)


# 2.01 DATA PRELOADING
print("\n\n2.01 DATA PRELOADING")

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
print(factor_scores_df.round(3))


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
print("\n\n2.08 COVERGENT VALIDITY")

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
print("\n\n2.09 DISCRIMINANT VALIDITY")

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
print("\n\n2.10 VALIDITY SUMMARY REPORT")

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


# Step03 Input Prior Data
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
      .drop("analyzedvar_Stock_Price_Rp").drop("Year").drop("Highest_Stock_Price_Rp").drop("Lowest_Stock_Price_Rp")
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


# 4.01 DATA PRELOADING
print("\n\n #4.01 DATA PRELOADING")

df = pd.read_csv("BBCA_analyzed.csv")

model_df = df[
    [
        "Year",
        "analyzedvar_Stock_Price_Rp",
        "analyzedvar_Dividends_Rp",
        "analyzedvar_ROA_pct",
        "analyzedvar_ROE_pct",
        "analyzedvar_Debt_to_Equity",
        "analyzedvar_EBITDA_BnRp",
    ]
].dropna()

print()
print(model_df)

# 4.02 SCATTER PLOT

# IV 1 =  DIVIDENDS 
df = pd.read_csv("BBCA_analyzed.csv")
df["Year"] = df["Year"].astype(str)

sns.scatterplot(
    data=df,
    x="analyzedvar_Dividends_Rp",
    y="analyzedvar_Stock_Price_Rp",
    hue="Year",
    palette="magma",
    s=70
)
plt.title("Stock Price vs Dividends")
plt.xlabel("Dividends (Rp)")
plt.ylabel("Stock Price (Rp)")
plt.grid (True, alpha=0.7)
plt.show()

# IV 2 =  ROA
df = pd.read_csv("BBCA_analyzed.csv")
df["Year"] = df["Year"].astype(str)

sns.scatterplot(
    data=df,
    x="analyzedvar_ROA_pct",
    y="analyzedvar_Stock_Price_Rp",
    hue="Year",
    palette="magma",
    s=70
)
plt.title("Stock Price vs ROA")
plt.xlabel("ROA (%)")
plt.ylabel("Stock Price (Rp)")
plt.grid (True, alpha=0.7)
plt.show()

# IV 3 =  ROE
df = pd.read_csv("BBCA_analyzed.csv")
df["Year"] = df["Year"].astype(str)

sns.scatterplot(
    data=df,
    x="analyzedvar_ROE_pct",
    y="analyzedvar_Stock_Price_Rp",
    hue="Year",
    palette="magma",
    s=70
)
plt.title("Stock Price vs ROE")
plt.xlabel("ROE (%)")
plt.ylabel("Stock Price (Rp)")
plt.grid (True, alpha=0.7)
plt.show()

# IV 4 =  DTE
df = pd.read_csv("BBCA_analyzed.csv")
df["Year"] = df["Year"].astype(str)

sns.scatterplot(
    data=df,
    x="analyzedvar_Debt_to_Equity",
    y="analyzedvar_Stock_Price_Rp",
    hue="Year",
    palette="magma",
    s=70
)
plt.title("Stock Price vs Debt to Equity")
plt.xlabel("Debt to Equity")
plt.ylabel("Stock Price (Rp)")
plt.grid (True, alpha=0.7)
plt.show()

# IV 5 =  EBITDA
df = pd.read_csv("BBCA_analyzed.csv")
df["Year"] = df["Year"].astype(str)

sns.scatterplot(
    data=df,
    x="analyzedvar_EBITDA_BnRp",
    y="analyzedvar_Stock_Price_Rp",
    hue="Year",
    palette="magma",
    s=70
)
plt.title("Stock Price vs EBITDA")
plt.xlabel("EBITDA")
plt.ylabel("Stock Price (Rp)")
plt.grid (True, alpha=0.7)
plt.show()


# 5.01 DATA PRELOADING
df = pd.read_csv("BBCA_analyzed.csv")
y = df["analyzedvar_Stock_Price_Rp"]
df["Year_c"] = df["Year"] - df["Year"].mean()

# 5.02A STOCK PRICE - DIVIDENDS
X1 = sm.add_constant(df[["analyzedvar_Dividends_Rp"]])
model1 = sm.OLS(y, X1).fit()
print("#5.02A STOCK PRICE - DIVIDENDS")
print(model1.summary())
# 5.02B STOCK PRICE - DIVIDENDS (YEAR AS CONTROL)
X2 = sm.add_constant(df[["analyzedvar_Dividends_Rp", "Year_c"]])
model2 = sm.OLS(y, X2).fit()
print("\n# 5.02B STOCK PRICE - DIVIDENDS (YEAR AS CONTROL)")
print(model2.summary())

# 5.03A STOCK PRICE - ROA
X3 = sm.add_constant(df[["analyzedvar_ROA_pct"]])
model1 = sm.OLS(y, X3).fit()
print("#5.03A STOCK PRICE - ROA")
print(model1.summary())
# 5.03B STOCK PRICE - ROA (YEAR AS CONTROL)
X4 = sm.add_constant(df[["analyzedvar_ROA_pct", "Year_c"]])
model2 = sm.OLS(y, X4).fit()
print("\n# 5.03B STOCK PRICE - ROA (YEAR AS CONTROL)")
print(model2.summary())

# 5.04A STOCK PRICE - ROE
X5 = sm.add_constant(df[["analyzedvar_ROE_pct"]])
model1 = sm.OLS(y, X5).fit()
print("#5.04A STOCK PRICE - ROE")
print(model1.summary())
# 5.04B STOCK PRICE - ROE (YEAR AS CONTROL)
X6 = sm.add_constant(df[["analyzedvar_ROE_pct", "Year_c"]])
model2 = sm.OLS(y, X6).fit()
print("\n# 5.04B STOCK PRICE - ROE (YEAR AS CONTROL)")
print(model2.summary())

# 5.05A STOCK PRICE - DTE
X7 = sm.add_constant(df[["analyzedvar_Debt_to_Equity"]])
model1 = sm.OLS(y, X7).fit()
print("#5.05A STOCK PRICE - DEBT TO EQUITY")
print(model1.summary())
# 5.05B STOCK PRICE - DTE (YEAR AS CONTROL)
X8 = sm.add_constant(df[["analyzedvar_Debt_to_Equity", "Year_c"]])
model2 = sm.OLS(y, X8).fit()
print("\n# 5.05B STOCK PRICE - DEBT TO EQUITY (YEAR AS CONTROL)")
print(model2.summary())

# 5.06A STOCK PRICE - EBITDA
X9 = sm.add_constant(df[["analyzedvar_EBITDA_BnRp"]])
model1 = sm.OLS(y, X9).fit()
print("#5.06A STOCK PRICE - EBITDA")
print(model1.summary())
# 5.06B STOCK PRICE - DTE (YEAR AS CONTROL)
X10 = sm.add_constant(df[["analyzedvar_EBITDA_BnRp", "Year_c"]])
model2 = sm.OLS(y, X10).fit()
print("\n# 5.06B STOCK PRICE - EBITDA (YEAR AS CONTROL)")
print(model2.summary())


# 5.07 P VALUE SUMMARY
df = pd.read_csv("BBCA_analyzed.csv")

y = df["analyzedvar_Stock_Price_Rp"]
X = df[[
    "analyzedvar_Dividends_Rp",
    "analyzedvar_EPS_Rp",
    "analyzedvar_PE",
    "analyzedvar_ROA_pct",
    "analyzedvar_ROE_pct",
    "analyzedvar_Debt_to_Equity",
    "analyzedvar_EBITDA_BnRp",
    "Year",
]]
X = sm.add_constant(X)

model = sm.OLS(y, X).fit()
print(model.summary())

coefs = model.params
std_err = model.bse
t_vals = model.tvalues
p_vals = model.pvalues
conf_int = model.conf_int(alpha=0.05)  # 95% CI

results_table = pd.DataFrame({
    "Variable": coefs.index,
    "Coefficient": coefs.values,
    "Std_Error": std_err.values,
    "t_value": t_vals.values,
    "p_value": p_vals.values,
    "CI_lower": conf_int[0].values,
    "CI_upper": conf_int[1].values,
})

results_table = results_table.round(3)
print(results_table)

results_table.to_csv("BBCA_regression_results.csv", index=False)


# 5.08 REGRESSION SUMMARY
rmse = np.sqrt(model.mse_resid)

print("Multiple R² =", model.rsquared)
print("Adjusted R² =", model.rsquared_adj)
print("F-statistic =", model.fvalue, "(p =", model.f_pvalue, ")")
print("n =", int(model.nobs))
print("RMSE=", rmse)


# 5.09 LINEARITY
fitted = model.fittedvalues
residuals = model.resid

plt.scatter(fitted, residuals)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs Fitted")
plt.show()

print(model.summary())