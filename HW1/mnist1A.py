from struct import unpack
import gzip
import numpy as np
from numpy import zeros, uint8, float32
import cv2
from scipy.stats import norm
from sklearn.ensemble import RandomForestClassifier
import time


def get_labeled_data(imagefile, labelfile):
    """Read input-vector (image) and target class (label, 0-9) and return
       it as list of tuples.
    """
    # Open the images with gzip in read binary mode
    images = gzip.open(imagefile, 'rb')
    labels = gzip.open(labelfile, 'rb')

    # Read the binary data

    # We have to get big endian unsigned int. So we need '>I'

    # Get metadata for images
    images.read(4)  # skip the magic_number
    number_of_images = images.read(4)
    number_of_images = unpack('>I', number_of_images)[0]
    rows = images.read(4)
    rows = unpack('>I', rows)[0]
    cols = images.read(4)
    cols = unpack('>I', cols)[0]

    # Get metadata for labels
    labels.read(4)  # skip the magic_number
    N = labels.read(4)
    N = unpack('>I', N)[0]
    # FOR TESTING
    # N = 1000 

    if number_of_images != N:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    x = zeros((N, rows, cols), dtype=uint8)  # Initialize numpy array
    y = zeros(N, dtype=uint8)  # Initialize numpy array
    print("STARTING " + str(N))
    for i in range(N):
        if i % 1000 == 0 and i != 0:
            print("i: %i" % i)
        tmp_label = labels.read(1)
        y[i] = unpack('>B', tmp_label)[0]
        pixel_count = 0
        for row in range(rows):
            for col in range(cols):
                tmp_pixel = images.read(1)  # Just a single byte
                tmp_pixel_prior = unpack('>B', tmp_pixel)[0]
                tmp_pixel = 1 if tmp_pixel_prior > 127 else 0
                x[i][row][col] = tmp_pixel
                pixel_count += 1
    return (x, y)


def get_bounding_border(img):
    x_min = -1
    x_max = -1
    y_min = -1
    y_max = -1
    rows = img.shape[0]
    cols = img.shape[1]
    for i in range(cols):
        sum_min = np.sum(img[:, i])
        if x_min == -1:
            if sum_min != 0:
                x_min = i

        sum_max = np.sum(img[:, - i - 1])
        if x_max == -1:
            if sum_max != 0:
                x_max = cols - i

    for i in range(rows):
        sum_min = np.sum(img[i, :])
        if y_min == -1:
            if sum_min != 0:
                y_min = i

        sum_max = np.sum(img[- i - 1, :])
        if y_max == -1:
            if sum_max != 0:
                y_max = rows - i
    return (x_min, x_max, y_min, y_max)


def get_index_for_label(digit_type):
    return np.where(train_labels[:] == digit_type)


def save_array_to_file(image_array, file_desc):
    print("SAVE ARRAYS")
    int_image_array = image_array.astype(uint8)
    for i in range(np.size(int_image_array, 0)):
        cv2.imwrite(file_desc + str(i) + ".png", int_image_array[i])


def find_pixel_bounding_boxes(image_array):
    print("FIND BOUNDING BOXES")
    resized_images = zeros((np.size(image_array, 0), 20, 20))
    for i in range(np.size(image_array, 0)):
        bounding_box_pixel_locs = get_bounding_border(image_array[i])

        bounded_image = image_array[i][bounding_box_pixel_locs[2]:bounding_box_pixel_locs[3],
                                       bounding_box_pixel_locs[0]:bounding_box_pixel_locs[1]].copy()
        resized = cv2.resize(bounded_image, (20, 20), interpolation=cv2.INTER_NEAREST)
        resized_images[i, :, :] = resized
    print("DONE")
    return resized_images


def count_images_per_label(labels):
    count_images = zeros((10, 1), dtype=uint8)
    for i in range(10):
        count_images[i, 0] = int(np.size(labels[np.where(labels[:] == i)], 0))
    return count_images


def calculate_mean_var_pixelcount_for_images(image_array, number_array):
    print("RECREATE MEANS AS BOX")
    size = np.size(image_array, 1)
    mean_image = np.zeros((10, size, size))
    var_image = np.zeros((10, size, size))
    count_pixel_image = np.zeros((10, size, size), dtype=uint8)
    for i in range(10):
        for row in range(size):
            for col in range(size):
                mean_image[i, row, col] = np.mean(image_array[get_index_for_label(i), row, col])
                var_image[i, row, col] = np.var(image_array[get_index_for_label(i), row, col])
                count_pixel_image[i, row, col] = int(np.sum(image_array[get_index_for_label(i), row, col]))
    print("DONE")
    return (mean_image, var_image, count_pixel_image)


def test_gaussian(test_array, labels, mean, var, image_counts):
    print("START TESTING GAUSSIAN")
    success = 0
    size = np.size(test_array, 1)
    total = np.size(test_array, 0)
    for image_number in range(total):
        test_image = test_array[image_number, :, :]
        if image_number % 1000 == 0:
            print("i: %i" % image_number)
        sum_array = zeros((10, 1))
        for num in range(10):
            for row in range(size):
                for col in range(size):
                    if var[num, row, col] != 0.0:
                        pixel_prob = norm.pdf(test_image[row, col], mean[num, row, col], np.sqrt(var[num, row, col]))
                        if pixel_prob != 0.0:
                            sum_array[num, 0] += np.log(pixel_prob)
                        else:
                            sum_array[num, 0] += 0.0
            sum_array[num, 0] += np.log(1.0*image_counts[num, 0]/total)
        if np.argmax(sum_array) == labels[image_number]:
            success += 1

    average_prob = (1.0 * success) / total
    print("AVERAGE PROBABILITY: " + str(average_prob))


def test_bernoulli(test_array, labels, pixel_counts, image_counts, total_train):
    print("START TESTING BERNOULLI")
    success = 0
    size = np.size(test_array, 1)
    total = np.size(test_array, 0)
    for image_number in range(total):
        sum_array = zeros((10, 1))
        if image_number % 1000 == 0 and image_number != 0:
            print("i: %i" % image_number)
        for num in range(10):
            for row in range(size):
                for col in range(size):
                    probability = (pixel_counts[num, row, col]*1.0)/(size*1.0 + 1)
                    if test_array[image_number, row, col] == 0:
                        probability = 1 - probability
                    if probability > 0.0:
                        sum_array[num, 0] += np.log(probability)
                    else:
                        sum_array[num, 0] += 0.0
            sum_array[num, 0] += np.log(image_counts[num, 0]/total_train)
        if np.argmax(sum_array) == labels[image_number]:
            success += 1

    average_prob = (1.0 * success) / total
    print("AVERAGE PROBABILITY: " + str(average_prob))


def predictions_with_classifier(train_array, train_labels, test_array, test_labels, estimators, max):
    classifier = RandomForestClassifier(n_jobs=-1, n_estimators=estimators, max_depth=max)
    classifier.fit(train_array, train_labels)
    predictions = classifier.predict(test_array)

    success = 0
    size_test_array = np.size(test_array, 0)
    for index in range(size_test_array):
        if predictions[index] == test_labels[index]:
            success += 1

    print("ACCURACY: " + str(success/size_test_array))


image_with_labels_train = get_labeled_data('train-images-idx3-ubyte.gz',
                                           'train-labels-idx1-ubyte.gz')

train_array = image_with_labels_train[0]
train_labels = image_with_labels_train[1]
count_images_train = count_images_per_label(train_labels)

bounded_images_original = find_pixel_bounding_boxes(train_array)
number_images_train = np.size(train_array, 0)

print("Calculate training data")
means_var_pixels_images_train = calculate_mean_var_pixelcount_for_images(train_array, train_labels)
means_image_train = means_var_pixels_images_train[0]
vars_image_train = means_var_pixels_images_train[1]
pixel_sums_train = means_var_pixels_images_train[2]

image_with_labels_test = get_labeled_data('t10k-images-idx3-ubyte.gz',
                                          't10k-labels-idx1-ubyte.gz')

test_array = image_with_labels_test[0]
test_labels = image_with_labels_test[1]

bounded_images_test_stretched = find_pixel_bounding_boxes(test_array)
number_images_test = np.size(test_array, 0)

print("Save means to file")
save_array_to_file(means_image * 255, "mean")

print("Test with train")
test_gaussian(train_array, train_labels, means_image_train, vars_image_train, count_images_train)
print("Test with test")
test_gaussian(test_array, test_labels, means_image_train, vars_image_train, count_images_train)

print("Test with train")
test_gaussian(bounded_images_original, train_labels, means_image_train_stretched, vars_image_train_stretched, count_images_train)
print("Test with test")
test_gaussian(bounded_images_test_stretched, test_labels, means_image_train_stretched, vars_image_train_stretched, count_images_train)


# Reshape for classifiers
test_array_reshape_original = np.reshape(test_array, (number_images_test, 784))
train_array_reshape_original = np.reshape(train_array, (number_images_train, 784))
test_array_reshape_stretched = np.reshape(bounded_images_test_stretched, (number_images_test, 400))
train_array_reshape_stretched = np.reshape(bounded_images_original, (number_images_train, 400))

print("10T 4D U")
predictions_with_classifier(train_array_reshape_original, train_labels, train_array_reshape_original, train_labels, 10, 4)
predictions_with_classifier(train_array_reshape_original, train_labels, test_array_reshape_original, test_labels, 10, 4)

print("10T 4D S")
predictions_with_classifier(train_array_reshape_stretched, train_labels, train_array_reshape_stretched, train_labels, 10, 4)
predictions_with_classifier(train_array_reshape_stretched, train_labels, test_array_reshape_stretched, test_labels, 10, 4)

print("10T 16D U")
predictions_with_classifier(train_array_reshape_original, train_labels, train_array_reshape_original, train_labels, 10, 16)
predictions_with_classifier(train_array_reshape_original, train_labels, test_array_reshape_original, test_labels, 10, 16)

print("10T 16D S")
predictions_with_classifier(train_array_reshape_stretched, train_labels, train_array_reshape_stretched, train_labels, 30, 16)
predictions_with_classifier(train_array_reshape_stretched, train_labels, test_array_reshape_stretched, test_labels, 30, 16)

print("30T 4D U")
predictions_with_classifier(train_array_reshape_original, train_labels, train_array_reshape_original, train_labels, 30, 4)
predictions_with_classifier(train_array_reshape_original, train_labels, test_array_reshape_original, test_labels, 30, 4)

print("30T 4D S")
predictions_with_classifier(train_array_reshape_stretched, train_labels, train_array_reshape_stretched, train_labels, 30, 4)
predictions_with_classifier(train_array_reshape_stretched, train_labels, test_array_reshape_stretched, test_labels, 30, 4)

print("30T 16D U")
predictions_with_classifier(train_array_reshape_original, train_labels, train_array_reshape_original, train_labels, 30, 16)
predictions_with_classifier(train_array_reshape_original, train_labels, test_array_reshape_original, test_labels, 30, 16)

print("30T 16D S")
predictions_with_classifier(train_array_reshape_stretched, train_labels, train_array_reshape_stretched, train_labels, 30, 16)
predictions_with_classifier(train_array_reshape_stretched, train_labels, test_array_reshape_stretched, test_labels, 30, 16)
