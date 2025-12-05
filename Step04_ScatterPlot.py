#
# Step04, 2025/12/06
# File: Step04_ScatterPlot.py
# Short description of the task
#

import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)  # show all columns
pd.set_option("display.width", 200)

# 4.01 DATA PRELOADING
print("4.01 DATA PRELOADING")

df = pd.read_csv("BBCA_analyzed.csv")

model_df = df[
    [
        "Year",
        "analyzedvar_Stock_Price_Rp",
        "analyzedvar_Dividends_Rp",
        "analyzedvar_ROA_pct",
        "analyzedvar_ROE_pct",
        "analyzedvar_Debt_to_Equity",
        "analyzedvar_EBITDA_BnRp",
    ]
].dropna()

print()
print(model_df)

# 4.02 SCATTER PLOT

# IV 1 =  DIVIDENDS 
df = pd.read_csv("BBCA_analyzed.csv")
df["Year"] = df["Year"].astype(str)

sns.scatterplot(
    data=df,
    x="analyzedvar_Dividends_Rp",
    y="analyzedvar_Stock_Price_Rp",
    hue="Year",
    palette="magma",
    s=70
)
plt.title("Stock Price vs Dividends")
plt.xlabel("Dividends (Rp)")
plt.ylabel("Stock Price (Rp)")
plt.grid (True, alpha=0.7)
plt.show()

# IV 2 =  ROA
df = pd.read_csv("BBCA_analyzed.csv")
df["Year"] = df["Year"].astype(str)

sns.scatterplot(
    data=df,
    x="analyzedvar_ROA_pct",
    y="analyzedvar_Stock_Price_Rp",
    hue="Year",
    palette="magma",
    s=70
)
plt.title("Stock Price vs ROA")
plt.xlabel("ROA (%)")
plt.ylabel("Stock Price (Rp)")
plt.grid (True, alpha=0.7)
plt.show()

# IV 3 =  ROE
df = pd.read_csv("BBCA_analyzed.csv")
df["Year"] = df["Year"].astype(str)

sns.scatterplot(
    data=df,
    x="analyzedvar_ROE_pct",
    y="analyzedvar_Stock_Price_Rp",
    hue="Year",
    palette="magma",
    s=70
)
plt.title("Stock Price vs ROE")
plt.xlabel("ROE (%)")
plt.ylabel("Stock Price (Rp)")
plt.grid (True, alpha=0.7)
plt.show()

# IV 4 =  DTE
df = pd.read_csv("BBCA_analyzed.csv")
df["Year"] = df["Year"].astype(str)

sns.scatterplot(
    data=df,
    x="analyzedvar_Debt_to_Equity",
    y="analyzedvar_Stock_Price_Rp",
    hue="Year",
    palette="magma",
    s=70
)
plt.title("Stock Price vs Debt to Equity")
plt.xlabel("Debt to Equity")
plt.ylabel("Stock Price (Rp)")
plt.grid (True, alpha=0.7)
plt.show()

# IV 5 =  EBITDA
df = pd.read_csv("BBCA_analyzed.csv")
df["Year"] = df["Year"].astype(str)

sns.scatterplot(
    data=df,
    x="analyzedvar_EBITDA_BnRp",
    y="analyzedvar_Stock_Price_Rp",
    hue="Year",
    palette="magma",
    s=70
)
plt.title("Stock Price vs EBITDA")
plt.xlabel("EBITDA")
plt.ylabel("Stock Price (Rp)")
plt.grid (True, alpha=0.7)
plt.show()