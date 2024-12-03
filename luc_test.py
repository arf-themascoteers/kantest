import pandas as pd

df = pd.read_csv("lucas.csv")
print(df.columns)

columns_to_keep = ['400'] + list(df.columns[1:-1][::(len(df.columns[1:-1]) // 15)])
columns_to_keep.append('oc')
downsampled_df = df[columns_to_keep]

downsampled_df.to_csv("lucas_min.csv", index=False)