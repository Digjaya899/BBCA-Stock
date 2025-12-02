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

z_scores = np.abs(stats.zscore(df [[
     "Year","Stock Price (Rp)","EPS (Rp)","Dividends (Rp)",
     "P/E","ROA (%)","ROE (%)","Debt-to-Equity",
     "Total Assets (Rp)","Total Liabilities (Rp)","Total Debt (Rp)","Total Equity (Rp)",
     "Revenue (Rp)","Net Profit (Rp)","Operating Cash Flow (Rp)",
    ]].describe().loc[["count","mean","std","min","max"]]
    .dropna()))

outlier_mask = z_scores > 3


# 3. Output