import pandas as pd

df = pd.read_csv("indian_pines.csv")
print(df.columns)
print(sorted(df['class'].unique().tolist()))