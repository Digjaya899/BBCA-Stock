#
# Step05, 2025/12/06
# File: Step05_RegressionAnalysis.py
# Short description of the task
#

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
### 5.02 R2 < CVR2 Difference under 0.10 (Acceptable) ; R2>0.30 (OK)

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
### 5.03 R2 < CVR2 Difference more than 0.10 (Concerning Overfitting) ; R2<0.30 (NOT OK)

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
### 5.04 R2 < CVR2 Difference more than 0.10 (Concerning Overfitting) ; R2<0.30 (NOT OK)

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
### 5.05 R2 < CVR2 Difference less than 0.10 (Acceptable) ; R2>0.30 (OK)

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
### 5.06 R2 < CVR2 Difference less than 0.10 (Acceptable) ; R2>0.30 (OK)