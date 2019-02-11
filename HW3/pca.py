import numpy as np
from numpy import zeros
from sklearn.decomposition import PCA


def calculate_mse_for_dataset(fit_dataset, transform_dataset, compared_dataset):
    tracking_mean_sqaure_error_table = zeros((5))
    for pca_index in range(5):
        pca = PCA(n_components=pca_index)
        pca.fit(fit_dataset)
        transformed_dataset = pca.transform(transform_dataset)
        inverse_transform = pca.inverse_transform(transformed_dataset)
        tracking_mean_sqaure_error_table[pca_index] = np.mean(np.subtract(inverse_transform, compared_dataset)**2) * compared_dataset.shape[1]
        print(f"{pca_index} {tracking_mean_sqaure_error_table[pca_index]}")
    return np.transpose(tracking_mean_sqaure_error_table)


dataset1 = np.genfromtxt("dataI.csv", delimiter=",", skip_header=1)
dataset2 = np.genfromtxt("dataII.csv", delimiter=",", skip_header=1)
dataset3 = np.genfromtxt("dataIII.csv", delimiter=",", skip_header=1)
dataset4 = np.genfromtxt("dataIV.csv", delimiter=",", skip_header=1)
dataset5 = np.genfromtxt("dataV.csv", delimiter=",", skip_header=1)
dataset_iris = np.genfromtxt("iris.csv", delimiter=",", skip_header=1)

datasets = [dataset1, dataset2, dataset3, dataset4, dataset5]

mean_square_error_table = zeros((5, 10))

dataset_index = 0
for dataset in datasets:
    noiseless_datasets = calculate_mse_for_dataset(dataset_iris, dataset, dataset_iris)
    noisey_datasets = calculate_mse_for_dataset(dataset, dataset, dataset_iris)
    mean_square_error_table[dataset_index, :5] = noiseless_datasets
    mean_square_error_table[dataset_index, 5:] = noisey_datasets
    dataset_index += 1
headers1 = "0N, 1N, 2N, 3N, 4N, 0c, 1c, 2c, 3c, 4c"
np.savetxt("renb2-numbers.csv", mean_square_error_table, delimiter=",", header=headers1, comments="")

pca = PCA(n_components=2)
pca.fit(dataset1)
dataset1_transformed = pca.transform(dataset1)
dataset1_inversed = pca.inverse_transform(dataset1_transformed)
headers2 = "Sepal.Length,Sepal.Width,Petal.Length,Petal.Width"
np.savetxt("renb2-recon.csv", dataset1_inversed, delimiter=",", header=headers2, comments="")
