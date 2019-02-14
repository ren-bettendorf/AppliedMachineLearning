import numpy as np
from numpy import zeros
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import euclidean_distances
import pickle
import json


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


# https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/cifar10.py
def separate_data_with_label(data):
    return data[b'data'], np.asarray(data[b'labels'])


# Data: [10000, 3072] 1024R 1024G 1024B
data1 = unpickle("data_batch_1")
data1_data, data1_labels = separate_data_with_label(data1)
data2 = unpickle("data_batch_2")
data2_data, data2_labels = separate_data_with_label(data2)
data3 = unpickle("data_batch_3")
data3_data, data3_labels = separate_data_with_label(data3)
data4 = unpickle("data_batch_4")
data4_data, data4_labels = separate_data_with_label(data4)
data5 = unpickle("data_batch_5")
data5_data, data5_labels = separate_data_with_label(data5)
test_data = unpickle("test_batch")
test_data, test_labels = separate_data_with_label(test_data)

with open('hw4_label_ordering.json') as json_data:
    label_json = json.load(json_data)

complete_dataset = np.concatenate((data1_data, data2_data, data3_data, data4_data, data5_data, test_data))
labels = np.concatenate((data1_labels, data2_labels, data3_labels, data4_labels, data5_labels, test_labels))

classifier_data = []
for i in range(10):
    classifier_data.append(complete_dataset[np.where(labels[:] == 0), :][0])

pca = PCA(n_components=20)
dataset_transformed = pca.fit_transform(classifier_data[0])
