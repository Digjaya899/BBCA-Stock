#
# Test, 2025/12/05
# File: Test.py
# Short description of the task
#
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt 
from scipy import stats 
from sklearn.decomposition import FactorAnalysis

# 1. Input
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

z_scores = np.abs(
    stats.zscore(
        df [["Year","Stock_Price_Rp","Highest_Stock_Price_Rp","Lowest_Stock_Price_Rp","Dividends_Rp",
               "EPS_Rp","PE","ROA_pct","ROE_pct","Debt_to_Equity",
               "Total_Assets_BnRp","Total_Liabilities_BnRp","Total_Debt_BnRp","Total_Equity_BnRp",
               "Revenue_BnRp","EBITDA_BnRp","Net_Profit_BnRp","Operating_Income_BnRp","Operating_Cost_BnRp",]
               ].dropna()))

outlier_mask = z_scores > 3

outliers = df.loc[outlier_mask.any(axis=1),
                  ["Year","Stock_Price_Rp","Highest_Stock_Price_Rp","Lowest_Stock_Price_Rp","Dividends_Rp",
                   "EPS_Rp","PE","ROA_pct","ROE_pct","Debt_to_Equity",
                   "Total_Assets_BnRp","Total_Liabilities_BnRp","Total_Debt_BnRp","Total_Equity_BnRp",
                   "Revenue_BnRp","EBITDA_BnRp","Net_Profit_BnRp","Operating_Income_BnRp","Operating_Cost_BnRp",],]

print(outliers)