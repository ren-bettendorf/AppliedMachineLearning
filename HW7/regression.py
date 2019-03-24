import pandas as pd
import string
from collections import Counter
import nltk

df = pd.read_csv("yelp_2k.csv", usecols=[3, 5])
# print(f"{df}")
df["text"] = df["text"].str.lower()
df["text"] = df["text"].str.replace(r'[^\w\s]+', '')
print(f"{df}")

# https://stackoverflow.com/a/46786277
word_dist = df["text"].str.split(expand=True).stack().value_counts()
print(f"{word_dist}")
