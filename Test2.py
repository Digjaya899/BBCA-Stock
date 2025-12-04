#
# Test2, 2025/12/04
# File: Test2.py
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

# 2. Process

# 3. Output
if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

# exp_cols = ["Year","Stock_Price_Rp","EPS_Rp","Dividends_Rp",
#             "PE","ROA_pct","ROE_pct","Debt_to_Equity",
#             "Total_Assets_BnRp","Total_Liabilities_BnRp","Total_Debt_BnRp","Total_Equity_BnRp",
#             "Revenue_BnRp","Net_Profit_BnRp","Operating_Cash_Flow_BnRp",]

# ["Year","Stock_Price_Rp","Highest_Stock_Price_Rp","Lowest_Stock_Price_Rp","Dividends_Rp",
#  "EPS_Rp","PE","ROA_pct","ROE_pct","Debt_to_Equity",
#  "Total_Assets_BnRp","Total_Liabilities_BnRp","Total_Debt_BnRp","Total_Equity_BnRp",
#  "Revenue_BnRp","EBITDA_BnRp","Net_Profit_BnRp","Operating_Income_BnRp","Operating_Cost_BnRp",]

exp_cols = ["Year","Stock_Price_Rp","Highest_Stock_Price_Rp","Lowest_Stock_Price_Rp","Dividends_Rp",
     "EPS_Rp","PE","ROA_pct","ROE_pct","Debt_to_Equity",
     "Total_Assets_BnRp","Total_Liabilities_BnRp","Total_Debt_BnRp","Total_Equity_BnRp",
     "Revenue_BnRp","EBITDA_BnRp","Net_Profit_BnRp","Operating_Income_BnRp","Operating_Cost_BnRp",]

#DebttoEquity, ROApct, PE, ROE, EPS, Dividends


if "fa_model" not in globals():
    fa_model = FactorAnalysis(n_components=3, random_state=0).fit(df[exp_cols])

loadings = pd.DataFrame(
    fa_model.components_.T,
    index=exp_cols,
    columns=["factor1", "factor2", "factor3"],)

communalities = (loadings**2).sum(axis=1)
validation_table = pd.DataFrame(
    {"communalities": communalities.round(3),
     "uniqueness": (1 - communalities).round(3),})

print(validation_table)