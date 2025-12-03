#
# Step01Data, 2025/12/01
# File: Step01Data.py
# Short description of the task
#
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Input
df = pd.read_csv ('BBCA.csv')

# 2A. Missing Data Analysis
print ("\n--- Data Types ---")
print(df.isnull().sum())
print(df.isnull().sum().sum())

#2B Missing Data Visualization
sns.heatmap(df.isnull())
plt.title("Missing Data Visualization")
plt.show()

# 3. Outlier Detection
columns = df.select_dtypes(include='number').columns 
for column in columns:
    sns.boxplot(x=df[column])
    plt.title(f"Outlier Checking: {column}")
    plt.show()