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

# 1.1 Load Data
df = pd.read_csv("BBCA.csv")
df [[
     "Year","Stock Price (Rp)","EPS (Rp)","Dividends (Rp)",
     "P/E","ROA (%)","ROE (%)","Debt-to-Equity",
     "Total Assets (Rp)","Total Liabilities (Rp)","Total Debt (Rp)","Total Equity (Rp)",
     "Revenue (Rp)","Net Profit (Rp)","Operating Cash Flow (Rp)",
    ]].head()
print (df)

# 1.2 Data Quality Checking
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

summary = df [[
     "Year","Stock Price (Rp)","EPS (Rp)","Dividends (Rp)",
     "P/E","ROA (%)","ROE (%)","Debt-to-Equity",
     "Total Assets (Rp)","Total Liabilities (Rp)","Total Debt (Rp)","Total Equity (Rp)",
     "Revenue (Rp)","Net Profit (Rp)","Operating Cash Flow (Rp)",
    ]].describe().loc[["count","mean","std","min","max"]]
print(summary)

# 1.3 Missing Data Analysis
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

missing_counts = df.isna().sum()
missing_counts[missing_counts>0]
print(missing_counts)

# 1.4 Outlier Detection
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

z_scores = np.abs(
    stats.zscore(
        df [["Stock Price (Rp)","EPS (Rp)","Dividends (Rp)",
             "P/E","ROA (%)","ROE (%)","Debt-to-Equity",]]
            .dropna()
        )
    )

outlier_mask = z_scores > 3

outliers = df.loc[outlier_mask.any(axis=1),
                  ["Year","Stock Price (Rp)","EPS (Rp)","Dividends (Rp)",
                   "P/E","ROA (%)","ROE (%)","Debt-to-Equity",],
                   ]

print(outliers)

# 1.5 Reliability Testing (Cronbach's Alpha)
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

service_items = df[
    ["Year","Stock Price (Rp)","EPS (Rp)","Dividends (Rp)",
     "P/E","ROA (%)","ROE (%)","Debt-to-Equity",
     "Total Assets (Rp)","Total Liabilities (Rp)","Total Debt (Rp)","Total Equity (Rp)",
     "Revenue (Rp)","Net Profit (Rp)","Operating Cash Flow (Rp)",]
]

item_var = service_items.var(axis=0, ddof=1)
total_var = service_items.sum(axis=1).var(ddof=1)

k = len(service_items.columns)
alpha = k / (k - 1) * (1 - item_var.sum() / total_var)

print("Cronbach's Alpha:", round(alpha, 3))

#1.6 Validation Checklist
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

summary = pd.Series(
    {
        "rows": df.shape[0],
        "features": df.shape[1],
        "numeric_columns": int(df.select_dtypes(include=["number"]).shape[1]),
        "rows_with_missing": int(df.isna().any(axis=1).sum()),
        "int_columns": int((df.dtypes == "int64").sum()),
        "float_columns": int((df.dtypes == "float64").sum()),
    }
)

print(summary)


# 3. Output