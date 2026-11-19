
import pandas as pd
df = pd.read_csv("data/creditcard.csv")

print("Shape:", df.shape)
print("Columns:", list(df.columns))
print("Fraud cases:", df["Class"].sum())
print("Fraud rate: {:.4%}".format(df["Class"].mean()))
print()
print(df.head())
print()
print(df["Amount"].describe())