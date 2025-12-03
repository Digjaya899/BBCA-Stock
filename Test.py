#
# Test, 2025/12/02
# File: Test.py
# Short description of the task
#
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy import stats 

# 1. Input
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

# 2. Process

# Step02 MetaData

# if "df" not in globals():
#     df = pd.read_csv("BBCA.csv")

# exp_cols = ["Year","Stock Price (Rp)","EPS (Rp)","Dividends (Rp)",
#      "P/E","ROA (%)","ROE (%)","Debt-to-Equity",
#      "Total Assets (Rp)","Total Liabilities (Rp)","Total Debt (Rp)","Total Equity (Rp)",
#      "Revenue (Rp)","Net Profit (Rp)","Operating Cash Flow (Rp)",]


# ["Year","Stock_Price_Rp","EPS_Rp","Dividends_Rp",
# "PE","ROA_pct","ROE_pct","Debt_to_Equity",
# "Total_Assets_BnRp","Total_Liabilities_BnRp","Total_Debt_BnRp","Total_Equity_BnRp",
# "Revenue_BnRp","Net_Profit_BnRp","Operating_Cash_Flow_BnRp",]