import numpy as np
import matplotlib.pyplot as plt
from numpy import zeros, int8, uint8, float32
import os
import math

POSITIVE = ">50K"
NEGATIVE = "<=50K"
FILE_NAME = "submission.txt"


def generate_scaled_array(array_in):
    scaled_array = np.zeros((array_in.shape[0], array_in.shape[1]), dtype=np.float)
    for i in range(np.size(array_in, 1)):
        col = array_in[:, i]
        scaled_array[:, i] = (col - np.mean(col)) / np.std(col)
    return scaled_array


def write_predictions_to_file(predictions_array):
    if os.path.isfile(FILE_NAME):
        print(f"Deleting old {FILE_NAME}")
        os.remove(FILE_NAME)
    with open(FILE_NAME, "w+") as file:
        for row in range(predictions_array.shape[0]):
            if predictions_array[row] == 1:
                file.write(POSITIVE + "\n")
            else:
                file.write(NEGATIVE + "\n")


def update_gradient(a, b, regularization, learning_rate, x_i, y_i):
    if y_i * (np.dot(a, x_i) + b) >= 1:
        next_a = a - (learning_rate * regularization * a)
        next_b = b
    else:
        next_a = a - (learning_rate * (regularization * a - y_i * x_i))
        next_b = b - (learning_rate * -y_i)
    return next_a, next_b


def predict(a, b, x):
    y_predicted = np.zeros((x.shape[0]), dtype=np.float32)
    signed_predictions = np.zeros((x.shape[0]), dtype=int8)
    for i in range(x.shape[0]):
        y_predicted[i] = np.dot(np.transpose(a), x[i, :]) + b
    signed_predictions[np.where(y_predicted[:] <= 0)] = -1
    signed_predictions[np.where(y_predicted[:] > 0)] = 1
    return signed_predictions


def calculate_accuracy(predicted, actual):
    success = 0
    for i in range(predicted.shape[0]):
        if predicted[i] == actual[i]:
            success += 1
    return 1.0 * success / predicted.shape[0]


training_array_total = np.genfromtxt('training_copied.csv', missing_values='', dtype='str', delimiter=',')
training_array_total = training_array_total[1:, 1:]
training_classifiers_before = training_array_total[:, -1]
training_classifiers = np.zeros((np.size(training_classifiers_before), 1), dtype=np.int8)
for i in range(np.size(training_classifiers_before, 0)):
    training_classifiers[i] = 1 if training_classifiers_before[i] == POSITIVE else -1
training_array = training_array_total[:, [0, 2, 4, 10, 11, 12]].copy().astype(np.float)

testing_array_total = np.genfromtxt('testing_copied.csv', missing_values='', dtype='str', delimiter=',')
testing_array_total = testing_array_total[1:, 1:]
testing_classifiers_before = testing_array_total[:, -1]
testing_classifiers = np.zeros((np.size(testing_classifiers_before), 1), dtype=np.int8)
for i in range(np.size(testing_classifiers_before, 0)):
    testing_classifiers[i] = 1 if testing_classifiers_before[i] == POSITIVE else -1
testing_array = testing_array_total[:, [0, 2, 4, 10, 11, 12]].copy().astype(np.float)


# https://stackoverflow.com/questions/31152967/normalise-2d-numpy-array-zero-mean-unit-variance
scaled_training = generate_scaled_array(training_array)
scaled_testing = generate_scaled_array(testing_array)

# https://stackoverflow.com/questions/3674409/how-to-split-partition-a-dataset-into-training-and-test-datasets-for-e-g-cros/37193593
number_rows = scaled_training.shape[0]
random_indices = np.random.permutation(number_rows)
training_indices = random_indices[:int(number_rows * .9)]
validation_indices = random_indices[int(number_rows * .9):]

x_training = scaled_training[training_indices, :]
x_validation = scaled_training[validation_indices, :]
y_training = training_classifiers[training_indices]
y_validation = training_classifiers[validation_indices]


regularization_constants = [1e-3, 1e-2, 1e-1, 1]
m = 1
n = 20
seasons = 50
steps = 300
update_accuracy_steps = 30
number_holdout = 50

best_a = np.zeros(6)
best_b = 0
best_reg = 0
best_average = 0.0
size_plots = int(steps * seasons / update_accuracy_steps)
plot_running_averages = zeros((len(regularization_constants), size_plots))
plot_magnitude = zeros((len(regularization_constants), size_plots))
reg_con_index = 0
print("Starting search for best reg")
for reg_con in regularization_constants:
    a = np.zeros(6)
    b = 1.0
    step_index = 0
    for season in range(1, seasons + 1):
        # https://towardsdatascience.com/learning-rate-schedules-and-adaptive-learning-rate-methods-for-deep-learning-2c8f433990d1
        # learning_rate_season = m / n * math.pow(0.5, math.floor(season / 4))
        # learning_rate_season = m / (n + 0.1 * season)
        learning_rate_season = m / n
        if season % 5 == 0:
            print(f"SEASON {season} rate {learning_rate_season} reg {reg_con}")
        random_indices_season = np.random.permutation(x_training.shape[0])
        season_holdout_indices = random_indices_season[:number_holdout]
        season_training_indices = random_indices_season[number_holdout:]
        holdout_set_x = x_training[season_holdout_indices, :]
        holdout_set_y = y_training[season_holdout_indices, :]
        training_set_x = x_training[season_training_indices, :]
        training_set_y = y_training[season_training_indices, :]
        for step in range(1, steps + 1):
            random_index = np.random.choice(range(training_set_x.shape[0]), 1)[0]
            x_i = training_set_x[random_index, :]
            y_i = training_set_y[random_index]
            a, b = update_gradient(a, b, reg_con, learning_rate_season, x_i, y_i)

            if (step % update_accuracy_steps == 0) and (step > 0):
                predictions = predict(a, b, holdout_set_x)
                seasons_accuracy = calculate_accuracy(predictions, holdout_set_y)
                plot_running_averages[reg_con_index, step_index] = seasons_accuracy
                plot_magnitude[reg_con_index, step_index] = np.linalg.norm(a)
                # print(f"STEP {step} ACCURACY {seasons_accuracy})
                step_index += 1
    validation_predictions = predict(a, b, x_validation)
    seasons_accuracy = calculate_accuracy(validation_predictions, y_validation)
    if seasons_accuracy > best_average:
        best_average = seasons_accuracy
        best_reg = reg_con
        best_a = a
        best_b = b
    reg_con_index += 1


print(f"A {best_a} B {best_b} REG {best_reg} AVE {best_average}")

x_axis = np.arange(1, size_plots + 1)
plt.figure(1)
plt.plot(x_axis, plot_magnitude[0, :])
plt.plot(x_axis, plot_magnitude[1, :])
plt.plot(x_axis, plot_magnitude[2, :])
plt.plot(x_axis, plot_magnitude[3, :])
plt.legend(['1e-3', '1e-2', '1e-1', '1'], loc='best')
plt.xlabel("Steps")
plt.ylabel("Magnitude")
plt.ylim(bottom=0, top=2.5)


plt.figure(2)
plt.plot(x_axis, plot_running_averages[0, :])
plt.plot(x_axis, plot_running_averages[1, :])
plt.plot(x_axis, plot_running_averages[2, :])
plt.plot(x_axis, plot_running_averages[3, :])

plt.legend(['1e-3', '1e-2', '1e-1', '1'], loc='best')
plt.xlabel("Steps")
plt.ylabel("Accuracy")
plt.ylim(bottom=0, top=1)
plt.show()

test_predictions = predict(best_a, best_b, scaled_testing)
write_predictions_to_file(test_predictions)
predicted_accuracy = calculate_accuracy(test_predictions, testing_classifiers)

print(f"ACCURACY {predicted_accuracy}")
