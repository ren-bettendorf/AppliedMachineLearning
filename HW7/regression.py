import pandas as pd
import matplotlib.pyplot as plt
import string
from nltk.probability import FreqDist

df = pd.read_csv("yelp_2k.csv", usecols=[3, 5])
# print(f"{df}")
df["text"] = df["text"].str.lower()
df["text"] = df["text"].str.replace(r'[^\w\s]+', '')
df["text"] = df["text"].str.replace('\r\n', '')
df["text"] = df["text"].str.replace('\r', '')
df["text"] = df["text"].str.replace('\r\n\r', '')
df["text"] = df["text"].str.replace('\r\n\r\n', '')

split_words = df["text"].str.split(" ")
total_words_list = []
for review in split_words:
    for word in review:
        # Remove empty words
        if word:
            total_words_list.append(word)
print(f"{len(total_words_list)}")
freq_dist = FreqDist(total_words_list)
print(f"{len(freq_dist)}")

word_counts = [word[1] for word in freq_dist.most_common()]
# plt.figure(1)
# plt.plot(word_counts)
# plt.title("Word Frequency")
# plt.xlabel("Word Rank")
# plt.ylabel("Word Count")
# plt.show()

# After analysis of the image I decided to remove 25 which is roughly when words 
# Remove words that only occur once
stop_words_map = freq_dist.most_common(25)
stop_words = freq_dist.hapaxes()
for stop_word in stop_words_map:
    stop_words.append(stop_word[0])

cleaned_word_list = [word for word in total_words_list if word not in stop_words]
print(f"{len(cleaned_word_list)}")
plt.figure(2)
plt.plot(word_counts)
plt.title("Word Frequency")
plt.xlabel("Word Rank")
plt.ylabel("Word Count")
plt.show()