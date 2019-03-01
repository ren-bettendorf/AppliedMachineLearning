import numpy as np
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from scipy.cluster.vq import vq
import matplotlib.pyplot as plt
import glob

CLASSIFIER_LABELS = ["Brush_Teeth", "Climb_Stairs", "Comb_Hair", "Descend_Stairs",
                     "Drink_Glass", "Eat_Meat", "Eat_Soup", "Getup_Bed", "Liedown_Bed",
                     "Pour_Water", "Sitdown_Chair", "Standup_Chair", "Use_Telephone", "Walk"]


def load_data():
    labels_data = {}
    for label in CLASSIFIER_LABELS:
        file_list = glob.glob("HMP_Dataset/" + label + "/*.txt")
        label_data = []
        for file in file_list:
            file_data = np.genfromtxt(file, delimiter=" ")
            label_data.append(file_data)
        labels_data[label] = label_data
    return labels_data


def create_data_segments(data, size_of_segments):
    size_of_segments = size_of_segments * 3
    resized_data = data[:len(data) - len(data) % size_of_segments]
    return np.reshape(resized_data, (-1, size_of_segments))


def combine_data_along_with_segment(size_of_segments, label_data, file_labels):
    total_data_segmented = []
    labels = []
    file_ids = []
    cluster_count = 0
    label_index = 0
    for label in file_labels:
        for i in range(len(label_data[label])):
            data = np.array(label_data[label][i].flatten())
            segmented_data_per_file = create_data_segments(
                data, size_of_segments)

            for j in range(segmented_data_per_file.shape[0]):
                file_ids.append(cluster_count)
                labels.append(file_labels[label_index])
            cluster_count += 1

            total_data_segmented.extend(segmented_data_per_file)
        label_index += 1
    data_segmented_numpy = np.array(total_data_segmented)
    file_ids_numpy = np.array(file_ids)
    labels_numpy = np.array(labels)
    return data_segmented_numpy, labels_numpy, file_ids_numpy


def fold_data(data):
    fold_indices = []
    kf = KFold(n_splits=3, shuffle=True)
    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    for train_index, test_index in kf.split(data):
        fold_indices.append(np.array((train_index, test_index)))
    return np.array(fold_indices)


def create_histograms(n_clusters, k_means, data, labels, file_ids):
    unique, classifier_counts = np.unique(file_ids, return_counts=True)
    feature_data = np.zeros((classifier_counts.shape[0], n_clusters), dtype=int)
    feature_label = np.zeros((classifier_counts.shape[0], 1), dtype=str)
    index = 0
    for i in range(classifier_counts.shape[0]):
        assignmentArr = np.array(vq(data[index:index + classifier_counts[i]], k_means.cluster_centers_)[0])
        for j in assignmentArr:
            feature_data[i][j] += 1
        feature_label[i] = labels[index]
        index = index + classifier_counts[i]
    return feature_data, feature_label


def create_histograms_for_predictions(n_cluster, train_data, train_labels, train_file_ids, test_data, test_labels, test_file_ids):
    k_means = KMeans(n_clusters=n_cluster, random_state=0).fit(train_data)
    train_histograms, train_labels_histogram = create_histograms(
        n_cluster, k_means, train_data, train_labels, train_file_ids)
    test_histograms, test_labels_histogram = create_histograms(
        n_cluster, k_means, test_data, test_labels, test_file_ids)


labeled_data = load_data()
data_segmented, labeled_data, file_ids = combine_data_along_with_segment(
    32, labeled_data, CLASSIFIER_LABELS)
folded_data_indices = fold_data(data_segmented)
for fold in folded_data_indices:
    train_data = data_segmented[fold[0]]
    train_labels = labeled_data[fold[0]]
    train_ids = file_ids[fold[0]]
    test_data = data_segmented[fold[1]]
    test_labels = labeled_data[fold[1]]
    test_ids = file_ids[fold[1]]
    create_histograms_for_predictions(
        20, train_data, train_labels, train_ids, test_data, test_labels, test_ids)
