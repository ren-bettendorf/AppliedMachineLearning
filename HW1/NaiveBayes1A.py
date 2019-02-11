import numpy as np
import scipy
from scipy.stats import norm


def get_mean_from_array(array, index):
    return np.mean(array[:, index])


def get_variance_from_array(array, index):
    return np.var(array[:, index])


data = np.genfromtxt("diabetes.csv", delimiter=",", skip_header=1)
data = data[1:, :]
rows = np.size(data, 0)
number_rows_training = int(rows * .8)
number_rows_testing = rows - number_rows_training

total_prob = 0.
for number_fold in range(10):
    np.random.shuffle(data)
    training = data[:number_rows_training, :]
    test = data[number_rows_training:, :]

    # Setup training data
    training_pos = training[np.where(training[:, 8] == 1)]
    training_neg = training[np.where(training[:, 8] == 0)]

    total_training = len(training_neg) + len(training_pos)
    prob_pos = 1.0*len(training_pos)/(len(training_pos) + len(training_neg))
    prob_neg = 1 - prob_pos

    # Setup test data
    test_pos = test[np.where(test[:, 8] == 1)]
    test_neg = test[np.where(test[:, 8] == 0)]
    total_test = len(test_neg) + len(test_pos)

    # Pregnancy
    preg_mean_pos = get_mean_from_array(training_pos, 0)
    preg_var_pos = get_variance_from_array(training_pos, 0)
    preg_mean_neg = get_mean_from_array(training_neg, 0)
    preg_var_neg = get_variance_from_array(training_neg, 0)

    # Glucose
    gluc_mean_pos = get_mean_from_array(training_pos, 1)
    gluc_var_pos = get_variance_from_array(training_pos, 1)
    gluc_mean_neg = get_mean_from_array(training_neg, 1)
    gluc_var_neg = get_variance_from_array(training_neg, 1)

    # BloodPressure
    bp_mean_pos = get_mean_from_array(training_pos, 2)
    bp_var_pos = get_variance_from_array(training_pos, 2)
    bp_mean_neg = get_mean_from_array(training_neg, 2)
    bp_var_neg = get_variance_from_array(training_neg, 2)

    # SkinThickness
    skin_mean_pos = get_mean_from_array(training_pos, 3)
    skin_var_pos = get_variance_from_array(training_pos, 3)
    skin_mean_neg = get_mean_from_array(training_neg, 3)
    skin_var_neg = get_variance_from_array(training_neg, 3)

    # Insulin
    insulin_mean_pos = get_mean_from_array(training_pos, 4)
    insulin_var_pos = get_variance_from_array(training_pos, 4)
    insulin_mean_neg = get_mean_from_array(training_neg, 4)
    insulin_var_neg = get_variance_from_array(training_neg, 4)

    # BMI
    bmi_mean_pos = get_mean_from_array(training_pos, 5)
    bmi_var_pos = get_variance_from_array(training_pos, 5)
    bmi_mean_neg = get_mean_from_array(training_neg, 5)
    bmi_var_neg = get_variance_from_array(training_neg, 5)

    # Diabetes Pedigree Function
    diabetes_mean_pos = get_mean_from_array(training_pos, 6)
    diabetes_var_pos = get_variance_from_array(training_pos, 6)
    diabetes_mean_neg = get_mean_from_array(training_neg, 6)
    diabetes_var_neg = get_variance_from_array(training_neg, 6)

    # Age
    age_mean_pos = get_mean_from_array(training_pos, 7)
    age_var_pos = get_variance_from_array(training_pos, 7)
    age_mean_neg = get_mean_from_array(training_neg, 7)
    age_var_neg = get_variance_from_array(training_neg, 7)

    correct = 0
    for i in range(number_rows_testing):
        normed_sum_pos = np.log(norm.pdf(test[i, 0], preg_mean_pos, np.sqrt(preg_var_pos))) + np.log(norm.pdf(test[i, 1], gluc_mean_pos, np.sqrt(gluc_var_pos))) + np.log(norm.pdf(test[i, 2], bp_mean_pos, np.sqrt(bp_var_pos))) + np.log(norm.pdf(test[i, 3], skin_mean_pos, np.sqrt(skin_var_pos))) + np.log(norm.pdf(test[i, 4], insulin_mean_pos, np.sqrt(insulin_var_pos))) + np.log(norm.pdf(test[i, 5], bmi_mean_pos, np.sqrt(bmi_var_pos))) + np.log(norm.pdf(test[i, 6], diabetes_mean_pos, np.sqrt(diabetes_var_pos))) + np.log(norm.pdf(test[i, 7], age_mean_pos, np.sqrt(age_var_pos)))
        normed_sum_neg = np.log(norm.pdf(test[i, 0], preg_mean_neg, np.sqrt(preg_var_neg))) + np.log(norm.pdf(test[i, 1], gluc_mean_neg, np.sqrt(gluc_var_neg))) + np.log(norm.pdf(test[i, 2], bp_mean_neg, np.sqrt(bp_var_neg))) + np.log(norm.pdf(test[i, 3], skin_mean_neg, np.sqrt(skin_var_neg)))  + np.log(norm.pdf(test[i, 4], insulin_mean_neg, np.sqrt(insulin_var_neg))) + np.log(norm.pdf(test[i, 5], bmi_mean_neg, np.sqrt(bmi_var_neg)))  + np.log(norm.pdf(test[i, 6], diabetes_mean_neg, np.sqrt(diabetes_var_neg))) + np.log(norm.pdf(test[i, 7], age_mean_neg, np.sqrt(age_var_neg)))
        prob_pos_test = normed_sum_pos + np.log(prob_pos)
        prob_neg_test = normed_sum_neg + np.log(prob_neg)

        if prob_pos_test > prob_neg_test:
            if test[i, 8] == 1:
                correct += 1
        else:
            if test[i, 8] == 0:
                correct += 1
    correct_probability = correct / number_rows_testing
    total_prob += correct_probability
    print("Iteration: " + str(number_fold + 1) + " Fraction Correct: " + str(correct) + " / " + str(number_rows_testing) + " \u221D " + str(correct / number_rows_testing))

average_prob = total_prob / 10
print("AVERAGE PROBABILITY: " + str(average_prob))
