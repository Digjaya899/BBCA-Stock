import pandas as pd
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

if "df" not in globals():
    df = pd.read_csv("BBCA_analyzed.csv")

iv_cols = [
    "analyzedvar_Dividends_Rp",
    "analyzedvar_EPS_Rp",
    "analyzedvar_PE",
    "analyzedvar_ROA_pct",
    "analyzedvar_ROE_pct",
    "analyzedvar_Debt_to_Equity",
    "analyzedvar_EBITDA_BnRp",
    # optionally add "Year" if you treat it as a regressor
]

X = df[iv_cols].dropna()
X_const = sm.add_constant(X)



vif_rows = []
for i, col in enumerate(X_const.columns):
    if col == "const":
        continue
    vif_value = variance_inflation_factor(X_const.values, i)
    vif_rows.append({"Variable": col, "VIF": round(vif_value, 2)})

vif_table = pd.DataFrame(vif_rows)
print(vif_table)
vif_table.to_csv("BBCA_vif_all_IV.csv", index=False)











import numpy as np

df2 = df.copy()
df2["ln_price"] = np.log(df2["analyzedvar_Stock_Price_Rp"])
df2["ln_EBITDA"] = np.log(df2["analyzedvar_EBITDA_BnRp"])

X_log = df2[["ln_EBITDA", "analyzedvar_Debt_to_Equity", "Year"]].dropna()
y_log = df2.loc[X_log.index, "ln_price"]

X_log_const = sm.add_constant(X_log)
model_log = sm.OLS(y_log, X_log_const).fit()
print(model_log.summary())
