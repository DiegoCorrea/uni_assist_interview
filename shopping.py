import csv
import sys
import calendar
from math import sqrt

from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix

TEST_SIZE = 0.4


class KNeighborsClassifier:

    def __init__(self, n_neighbors: int = 1):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train) -> None:
        self.X_train = X_train
        self.y_train = y_train

    @staticmethod
    def euclidean(p_1: list, p_2: list) -> float:
        return sqrt(sum(([(a - b) ** 2 for a, b in zip(p_1, p_2)])))

    def get_neighbors(self, x_test: list) -> list:
        distances = []
        for i, x_train in enumerate(self.X_train):
            distances.append((i, self.euclidean(x_train, x_test)))

        distances.sort(key=lambda x: x[1])

        return distances[:self.n_neighbors]

    def predict(self, X_test) -> np.array:
        """
        Predict labels
        """

        # placeholder
        predictions = []

        # go through examples
        for idx, x_test in enumerate(X_test):
            k_neighbors = self.get_neighbors(x_test)
            k_y_values = [self.y_train[i] for i, _ in k_neighbors]
            prediction = sum(k_y_values) / self.n_neighbors
            predictions.append(prediction)

        # return predictions
        return np.array(predictions)


def f1_score(Y_true: list, Y_pred: list) -> float:
    true_positive = sum(1 if y_true == 1 and y_pred == 1 else 0 for y_true, y_pred in zip(Y_true, Y_pred))
    false_positive = sum(1 if y_true == 0 and y_pred == 1 else 0 for y_true, y_pred in zip(Y_true, Y_pred))

    true_negative = sum(1 if y_true == 0 and y_pred == 0 else 0 for y_true, y_pred in zip(Y_true, Y_pred))
    false_negative = sum(1 if y_true == 1 and y_pred == 0 else 0 for y_true, y_pred in zip(Y_true, Y_pred))

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    return 2 * (precision * recall) / (precision + recall)


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)
    fi_score_value = f1_score(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

    print(f"F1 Score: {fi_score_value:.2f}")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    months = {month: index-1 for index, month in enumerate(calendar.month_abbr) if index}
    months['June'] = months.pop('Jun')

    evidence = []
    labels = []

    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            evidence.append([
                int(row['Administrative']),
                float(row['Administrative_Duration']),
                int(row['Informational']),
                float(row['Informational_Duration']),
                int(row['ProductRelated']),
                float(row['ProductRelated_Duration']),
                float(row['BounceRates']),
                float(row['ExitRates']),
                float(row['PageValues']),
                float(row['SpecialDay']),
                months[row['Month']],
                int(row['OperatingSystems']),
                int(row['Browser']),
                int(row['Region']),
                int(row['TrafficType']),
                1 if row['VisitorType'] == 'Returning_Visitor' else 0,
                1 if row['Weekend'] == 'TRUE' else 0
            ])
            labels.append(1 if row['Revenue'] == 'TRUE' else 0)

    return (evidence, labels)


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)

    return model



def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity = float(0)
    specificity = float(0)

    total_positive = float(0)
    total_negative = float(0)

    for label, prediction in zip(labels, predictions):

        if label == 1:
            total_positive += 1
            if label == prediction:
                sensitivity += 1

        if label == 0:
            total_negative += 1
            if label == prediction:
                specificity += 1

    sensitivity /= total_positive
    specificity /= total_negative

    return sensitivity, specificity


if __name__ == "__main__":
    main()