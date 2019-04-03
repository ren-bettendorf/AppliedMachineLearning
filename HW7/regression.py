import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from nltk.probability import FreqDist
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cosdis

df = pd.read_csv("yelp_2k.csv", usecols=[3, 5])
# print(f"{df}")
df["text_lower"] = df["text"].str.lower()
df["text_lower"] = df["text_lower"].str.replace(r'[^\w\s]+', '')
df["text_lower"] = df["text_lower"].str.replace('\r\n', '')
df["text_lower"] = df["text_lower"].str.replace('\r', '')
df["text_lower"] = df["text_lower"].str.replace('\r\n\r', '')
df["text_lower"] = df["text_lower"].str.replace('\r\n\r\n', '')
df["text_lower"] = df["text_lower"].str.replace('\n', '')
df["text_lower"] = df["text_lower"].str.replace('\n\n', '')

split_words = df["text_lower"].str.split(" ")
total_words_list = []
for review in split_words:
    for word in review:
        # Remove empty words
        if word:
            total_words_list.append(word)
print(f"TOTAL WORDS LENGTH: {len(total_words_list)}")
freq_dist = FreqDist(total_words_list)
print(f"FREQ DIST LENGTH: {len(freq_dist)}")

word_counts = [word[1] for word in freq_dist.most_common()]
# plt.figure(1)
# plt.plot(word_counts)
# plt.title("Word Frequency - Pre Processing")
# plt.xlabel("Word Rank")
# plt.ylabel("Word Count")
# plt.show()

# After analysis of the image
# decided to remove 25 which is roughly when words
# Remove words that only occur once
stop_words_map = freq_dist.most_common(25)
stop_words_list = freq_dist.hapaxes()
for stop_word in stop_words_map:
    stop_words_list.append(stop_word[0])

cv = CountVectorizer(stop_words=stop_words_list)
transformed_list = cv.fit_transform(df["text_lower"]).toarray()
# target = cv.transform(['Horrible customer service']).toarray()

# nn = NearestNeighbors(n_neighbors=20, metric='cosine')
# nn.fit(transformed_list)
# values, index = nn.kneighbors(target)
# values = values.flatten()
# index = index.flatten()
# count = 0
# print("--------TOP SCORES-------")
# for i in index:
#     review = df["text"][i][:200] if len(df["text"][i]) > 200 else df["text"][i]
#     print(f'Score: {values[count]} Review: {review}')
#     count += 1

# plt.figure(2)
# plt.plot(cleaned_word_counts)
# plt.title("Word Frequency - Post Processing")
# plt.xlabel("Word Rank")
# plt.ylabel("Word Count")
# plt.show()

star_array = df["stars"].values
train_text, test_text, train_star, test_star = train_test_split(transformed_list, star_array,
                                                                test_size=0.1, random_state=0)
lr = LogisticRegression()
lr.fit(train_text, train_star)
test_predictions = lr.predict(test_text)
