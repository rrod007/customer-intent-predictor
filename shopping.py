import csv
import sys
import random
import numpy
import pickle

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python shopping.py data [model.sav]")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    # print("Evidence: " + str(evidence) + "\n\n")
    # print("Labels: " + str(labels))

    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")

    # Save model
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        pickle.dump(model, open(filename, 'wb'))
        print(f"Model saved to {filename}")


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

    # set one list for all evidence lists, and another for all label values
    evidence = []
    labels = []

    # get each row from csv file into a *list of rows*
    with open(filename) as f:
        users = csv.reader(f, delimiter=',')

        # dictionaries to translate csv values to numerical values
        months = {"Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5, "Jul": 6,
                  "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11}
        user_type = {"New_Visitor": 0, "Returning_Visitor": 1, "Other": random.choice([0, 1])}
        weekend_revenue = {"TRUE": 1, "FALSE": 0}

        # iterate through each row
        first_row = True
        for user in users:
            # skip header row
            if first_row:
                first_row = False
                continue

            local_evidence = []

            # iterate through each evidence value and append it to local_evidence
            for i in range(len(user) - 1):
                # change val to numerical before appending if needed
                if i == 10:
                    local_evidence.append(months[user[i]])
                elif i == 15:
                    local_evidence.append(user_type[user[i]])
                elif i == 16:
                    local_evidence.append(weekend_revenue[user[i]])

                # change val type before appending if needed
                elif i == 0 or i == 2 or i == 4 or i == 11 or i == 12 or i == 13 or i == 14:
                    local_evidence.append(int(user[i]))
                elif i == 1 or i == 3 or i == 5 or i == 6 or i == 7 or i == 8 or i == 9:
                    local_evidence.append(float(user[i]))

            # append the complete list of evidence for the current user
            evidence.append(local_evidence)

            # append appropriate int value to labels for current user's label
            labels.append(weekend_revenue[user[-1]])

    return evidence, labels

    # raise NotImplementedError


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)

    return model

    # raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    true_positives = 0
    label_positives = 0

    true_negatives = 0
    label_negatives = 0

    for i in range(len(predictions)):
        if labels[i] == predictions[i] == 1:
            true_positives += 1
        if labels[i] == 1:
            label_positives += 1

        if labels[i] == predictions[i] == 0:
            true_negatives += 1
        if labels[i] == 0:
            label_negatives += 1

    return true_positives / label_positives, true_negatives / label_negatives

    # raise NotImplementedError


if __name__ == "__main__":
    main()
