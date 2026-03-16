import pandas as pd

df = pd.read_csv("data/raw/IFND.csv", encoding="latin1")

df = df[["Statement","Label"]]

df.columns = ["text","label"]

df["label"] = df["label"].apply(lambda x: 1 if x == "TRUE" else 0)  

df.to_csv("data/processed/indian_news_dataset.csv", index=False)

print(df.head())
print(df["label"].value_counts())