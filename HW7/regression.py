import pandas as pd
from nltk.corpus import stopwords
import string

stop = set(stopwords.words("english")) 
df = pd.read_csv("yelp_2k.csv", usecols=[3, 5])
# print(f"{df}")
df["text_length"] = df["text"].str.len()
# print(f"{df}")
# https://stackoverflow.com/questions/5486337/how-to-remove-stop-words-using-nltk-or-python
df["text_no_stopwords"] = df["text"].apply(lambda x: " ".join([word for word in x.split() if word not in stop]))
print(f"{df}")
df["text_no_stopwords"] = df["text_no_stopwords"].str.lower()
print(f"{df}")
df["text_no_stopwords"] = df["text_no_stopwords"].str.replace('[{}]'.format(string.punctuation), '')
print(f"{df}")