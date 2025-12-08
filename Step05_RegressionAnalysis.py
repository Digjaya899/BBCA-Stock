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

pd.set_option("display.max_columns", None)
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









# 5.07 P VALUE SUMMARY
df = pd.read_csv("BBCA_analyzed.csv")

y = df["analyzedvar_Stock_Price_Rp"]
X = df[[
    "analyzedvar_Dividends_Rp",
    "analyzedvar_ROA_pct",
    "analyzedvar_ROE_pct",
    "analyzedvar_Debt_to_Equity",
    "analyzedvar_EBITDA_BnRp",
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

