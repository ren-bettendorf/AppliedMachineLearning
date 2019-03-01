import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
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


labeled_data = load_data()
data_segmented, labeled_data, file_ids = combine_data_along_with_segment(
    32, labeled_data, CLASSIFIER_LABELS)
