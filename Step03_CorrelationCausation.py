#
# Step03, 2025/12/04
# File: Step03_CorrelationCausation.py
# Short description of the task
#
import pandas as pd

# 3.1 Purpose
import pandas as pd

if "df" not in globals():
    df = pd.read_csv("BBCA.csv")

target_corr = (
    df.corr(numeric_only=True)["Stock_Price_Rp"]
      .drop("Stock_Price_Rp").drop("Year")
      .sort_values(ascending=False)
      .head(5)
)

print(target_corr.round(3))

# 2. Process

# 3. Output