#
# Step05, 2025/12/06
# File: Step05_RegressionAnalysis.py
# Short description of the task
#

# "analyzedvar_Stock_Price_Rp",
# "analyzedvar_Dividends_Rp",
# "analyzedvar_ROA_pct",
# "analyzedvar_ROE_pct",
# "analyzedvar_Debt_to_Equity",
# "analyzedvar_EBITDA_BnRp",

import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)  # show all columns
pd.set_option("display.width", 200)

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
### 5.02 R2 < CVR2 Difference under 0.10 (Acceptable)

# 5.03A STOCK PRICE - ROA
X3 = sm.add_constant(df[["analyzedvar_ROA_pct"]])
model1 = sm.OLS(y, X3).fit()
print("#5.03A STOCK PRICE - ROA")
print(model1.summary())
# 5.03B STOCK PRICE - ROA (YEAR AS CONTROL)
X4 = sm.add_constant(df[["analyzedvar_ROA_pct", "Year_c"]])
model2 = sm.OLS(y, X4).fit()
print("\n# 5.02B STOCK PRICE - ROA (YEAR AS CONTROL)")
print(model2.summary())
### 5.03 R2 < CVR2 Difference more than 0.10 (Concerning Overfitting)

