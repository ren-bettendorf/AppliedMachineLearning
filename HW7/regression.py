import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from nltk.probability import FreqDist
from scikitplot.metrics import plot_roc
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cosdis


def report_predictions(predictions, actuals, isTest):
    correct = 0
    for prediction, actual in zip(predictions, actuals):
        if prediction == actual:
            correct += 1
    if isTest:
        print(f"Test Accuracy {correct/len(predictions)}")
    else:
        print(f"Train Accuracy {correct/len(predictions)}")

df = pd.read_csv("yelp_2k.csv", usecols=[3, 5])

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
freq_dist = FreqDist(total_words_list)

word_counts = [word[1] for word in freq_dist.most_common()]
plt.figure(1)
plt.plot(word_counts)
plt.title("Word Frequency - Pre Processing")
plt.xlabel("Word Rank")
plt.ylabel("Word Count")
plt.savefig("WordFrequency-Pre.png")
plt.clf()

stop_words_map = freq_dist.most_common(25)
stop_words_list = freq_dist.hapaxes()
for stop_word in stop_words_map:
    stop_words_list.append(stop_word[0])
# cleaned_words_list = []
# for word in total_words_list:
#     if word not in stop_words_list:
#         cleaned_words_list.append(word)
# cleaned_word_freqdist = FreqDist(cleaned_words_list)

# cleaned_word_counts = [word[1] for word in cleaned_word_freqdist.most_common()]
# plt.plot(cleaned_word_counts)
# plt.title("Word Frequency - Post Processing")
# plt.xlabel("Word Rank")
# plt.ylabel("Word Count")
# plt.savefig("WordFrequency-Post.png")
# plt.clf()

cv = CountVectorizer(stop_words=stop_words_list)
transformed_list = cv.fit_transform(df["text_lower"]).toarray()
horrible_customer_service = cv.transform(['Horrible customer service']).toarray()

nn = NearestNeighbors(n_neighbors=2000, metric='cosine')
nn.fit(transformed_list)
values, index = nn.kneighbors(horrible_customer_service)
values = values.flatten()
plt.plot(values)
plt.ylabel("Cos-Distance Score")
plt.xlabel("Cos-Distance Rank")
plt.savefig("Distance-Scores.png")
plt.clf()
index = index.flatten()
print("---------------")
for i in range(5):
    review = df["text"][index[i]][:200] if len(df["text"][[index[i]]]) > 200 else df["text"][[index[i]]]
    print(f'Score: {values[i]} Review: {review}')


star_array = df["stars"].values
train_text, test_text, train_star, test_star = train_test_split(transformed_list, star_array,
                                                                test_size=0.1, random_state=0)
lr = LogisticRegression()
lr.fit(train_text, train_star)
train_predictions = lr.predict(train_text)
report_predictions(train_predictions, train_star, False)
test_predictions = lr.predict(test_text)
report_predictions(test_predictions, test_star, True)

train_probability = lr.predict_proba(train_text)
positive_review = [probability[0] for probability in train_probability[train_predictions == 1]]
negative_review = [probability[0] for probability in train_probability[train_predictions == 5]]
plt.hist(positive_review, bins=100)
plt.hist(negative_review, bins=100)
plt.savefig("original_probability.png")
plt.clf()

test_probability = lr.predict_proba(test_text)
thresholds = [0.4, 0.55, 0.6, 0.65]
for threshold in thresholds:
    print(f"Starting threshold {threshold}")
    train_predictions = [1 if probability[0] > threshold else 5 for probability in train_probability]
    report_predictions(train_predictions, train_star, False)

    test_predictions = [1 if probability[0] > threshold else 5 for probability in test_probability]
    report_predictions(test_predictions, test_star, True)

plot_roc(test_star, test_probability)
plt.savefig("roc.png")
plt.clf()