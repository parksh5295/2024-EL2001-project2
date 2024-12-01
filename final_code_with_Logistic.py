import os
import sys
import argparse
import logging
from csv import reader
from random import randrange
from math import sqrt
from math import exp
from math import pi
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

# Convert string column to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for _ in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

# Evaluate an algorithm using a cross-validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

# Report the performance: accuracy, precision, recall
def report(predictions, answers):
    if len(predictions) != len(answers):
        logging.error("The lengths of two arguments should be the same")
        sys.exit(1)

    # accuracy
    accuracy = accuracy_score(answers, predictions)
    accuracy = round(accuracy * 100, 2)

    # precision
    precision = precision_score(answers, predictions)
    precision = round(precision * 100, 2)

    # recall
    recall = recall_score(answers, predictions)
    recall = round(recall * 100, 2)

    # f1-score
    f1 = f1_score(answers, predictions)
    f1 = round(f1 * 100, 2)

    logging.info(f"accuracy: {accuracy}%")
    logging.info(f"precision: {precision}%")
    logging.info(f"recall: {recall}%")
    logging.info(f"f1-score: {f1}%")

# Command-line arguments
def command_line_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--training", required=True, metavar="<file path to the training dataset>", help="File path of the training dataset", default="training.csv")
    parser.add_argument("-u", "--testing", required=True, metavar="<file path to the testing dataset>", help="File path of the testing dataset", default="testing.csv")
    parser.add_argument("-l", "--log", help="Log level (DEBUG/INFO/WARNING/ERROR/CRITICAL)", type=str, default="INFO")
    args = parser.parse_args()
    return args

# Logistic Regression Algorithm
def logistic_regression(train, test):
    train_X = [row[:-1] for row in train]  # Feature columns
    train_y = [row[-1] for row in train]  # Target column

    # Initialize and train the Logistic Regression model
    model = LogisticRegression(max_iter=200)
    model.fit(train_X, train_y)

    # Make predictions
    test_X = [row[:-1] for row in test]  # Feature columns
    predictions = model.predict(test_X)
    return predictions

# Main function
def main():
    args = command_line_args()
    logging.basicConfig(level=args.log)

    if not os.path.exists(args.training):
        logging.error("The training dataset does not exist: {}".format(args.training))
        sys.exit(1)

    if not os.path.exists(args.testing):
        logging.error("The testing dataset does not exist: {}".format(args.testing))
        sys.exit(1)

    # Load datasets
    train_dataset = load_csv(args.training)
    test_dataset = load_csv(args.testing)

    # Preprocess datasets
    for i in range(len(train_dataset[0]) - 1):
        str_column_to_float(train_dataset, i)
    str_column_to_int(train_dataset, len(train_dataset[0]) - 1)

    for i in range(len(test_dataset[0]) - 1):
        str_column_to_float(test_dataset, i)
    str_column_to_int(test_dataset, len(test_dataset[0]) - 1)

    # Run Logistic Regression and get predictions
    predictions = logistic_regression(train_dataset, test_dataset)

    # Report accuracy, precision, recall
    report(predictions, [row[-1] for row in test_dataset])

if __name__ == "__main__":
    main()
