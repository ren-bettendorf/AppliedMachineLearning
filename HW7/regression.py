import pandas as pd

df = pd.read_csv("yelp_2k.csv", usecols=[3, 5])
# print(f"{df}")
df["Review Length"] = df["text"].str.len()
print(f"{df}")
