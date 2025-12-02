#
# Step02, 2025/12/02
# File: Step02.py
# Short description of the task
#
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import factor_analyzer as fa 
from factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer import calculate_kmo

# 1. Input
df = pd.read_csv ('BBCA.csv') 
df_numbers = df.select_dtypes(include='number')

# 2. Bartlett
p_value= calculate_bartlett_sphericity(df_numbers) 
print(f"Bartlett's Test p value = {p_value}")

# 3. KMO
kmo_model = calculate_kmo(df_numbers)
print(f"KMO = {kmo_model}")

#FAIL!