import numpy as np
from numpy import zeros
import matplotlib.pyplot as plt
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


def save_mean_images(images):
    plt.figure(1)
    plt.subplots_adjust(hspace=0.35)
    for i in range(10):
        plt.subplot(4, 3, i + 1)
        # https://stackoverflow.com/questions/5086789/python-is-there-an-inverse-for-ndarray-flattenf
        rgb_image_flat = np.reshape(images[i], (3, -1)).T
        rgb_image_squared = np.reshape(rgb_image_flat, (32, 32, 3))
        plt.imshow(rgb_image_squared)
        plt.title(label_json[str(i)])
        plt.axis('off')
    plt.savefig("mean_images.png")
    plt.clf()


def create_classifier_data_and_mean_images(data, labels):
    classifier_data = []
    mean_classifier_images = np.zeros((10, 3072))
    for i in range(10):
        classifier_data.append(data[np.where(labels[:] == i), :][0])
        mean_classifier_images[i, :] = np.mean(classifier_data[i], axis=0)
    mean_classifier_images = mean_classifier_images.astype(int)
    return classifier_data, mean_classifier_images


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

labeled_names = []
for i in range(10):
    labeled_names.append(label_json[str(i)])

complete_dataset = np.concatenate((data1_data, data2_data, data3_data, data4_data, data5_data, test_data))
complete_labels = np.concatenate((data1_labels, data2_labels, data3_labels, data4_labels, data5_labels, test_labels))

classifier_data, mean_classifier_images = create_classifier_data_and_mean_images(complete_dataset, complete_labels)

# save_mean_images(mean_classifier_images)

# plt.figure(2)
# sum_squared_values = []
# for i in range(10):
#     pca = PCA(n_components=20)
#     pca.fit(classifier_data[i])
#     dataset_transformed = pca.transform(classifier_data[i])
#     inversed = pca.inverse_transform(dataset_transformed)
#     difference = classifier_data[i] - inversed
#     summed = np.sum(difference**2, axis=1)
#     sum_squared_error = np.mean(summed)
#     sum_squared_values.append(sum_squared_error)
    # print(f"ERROR: {sum_squared_error}")

# plt.figure(1, [10, 5])
# plt.bar(range(10), sum_squared_values, align='center')
# plt.xticks(range(10), labeled_names)
# plt.title("Sum Squared Error")
# plt.savefig('Sum_Squared_Error.png')
# plt.clf()

distance_matrix = euclidean_distances(mean_classifier_images, mean_classifier_images, squared=True)
np.savetxt('partb_distances.csv', distance_matrix, delimiter=',')
