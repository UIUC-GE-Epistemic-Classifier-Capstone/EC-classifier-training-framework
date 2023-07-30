import pandas as pd

# load your csv file
df = pd.read_csv("datasets/data/label_wrong labels.csv", header=None)
print(df.columns)
# reverse the binary labels in the first column (index 0)
df[0] = df[0].apply(lambda x: 1 if x==0 else 0)

# save your csv file
df.to_csv("label.csv", index=False, header=False)
