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
df[
    ["Year",
     "Stock_Price(Rp)",
     "EPS(Rp)","Dividends(Rp)",
     "ROE(%)",
     "Debt_to_Equity",
     "Total_Assets(Rp)",
     "Total_Liabilities(Rp)",
     "Total_Debt(Rp)",
     "Total_Equity(Rp)",
     "Revenue(Rp)",
     "Net_Profit(Rp)",
     "Operating_Cashflow(Rp)",
     "Comprehensive_Net_Profit(Rp)",
    ]
].head()
print(df)

# 1.2 Data Quality Checking
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

summary = df [[
     "Total_Assets(Rp)","Total_Liabilities(Rp)",
     "Total_Debt(Rp)","Total_Equity(Rp)",
     "Revenue(Rp)","Net_Profit(Rp)",
     "Operating_Cashflow(Rp)","Comprehensive_Net_Profit(Rp)",
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

z_scores = np.abs(stats.zscore(df[
    ["Year",
     "Stock_Price(Rp)",
     "EPS(Rp)","Dividends(Rp)",
     "ROE(%)",
     "Debt_to_Equity",
     "Total_Assets(Rp)",
     "Total_Liabilities(Rp)",
     "Total_Debt(Rp)",
     "Total_Equity(Rp)",
     "Revenue(Rp)",
     "Net_Profit(Rp)",
     "Operating_Cashflow(Rp)",
     "Comprehensive_Net_Profit(Rp)",
    ]].dropna()))

outlier_mask = z_scores > 3

df.loc

# 3. Output