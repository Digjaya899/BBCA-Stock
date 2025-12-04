#
# Step01AData, 2025/12/02
# File: Step01AData.py
# Short description of the task
#
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy import stats 

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
