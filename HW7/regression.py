import pandas as pd
import string

df = pd.read_csv("yelp_2k.csv", usecols=[3, 5])
# print(f"{df}")
df["text_length"] = df["text"].str.len()
# print(f"{df}")
# https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python