"""
Use previously trained models to predict whether a user is going to make a purchase or not
"""
import sys
import pickle
import csv
import random

classification_dict = {
    0: "Did not make a purchase",
    1: "Made a purchase"
}


def predict():
    if len(sys.argv) != 3:
        sys.exit("Usage: python predict.py model_to_use data.csv")

    # Load model
    model = pickle.load(open(sys.argv[1], 'rb'))

    # Load data
    users_data = []

    with open(sys.argv[2]) as f:
        users = csv.reader(f, delimiter=',')

        # dictionaries to translate csv values to numerical values
        months = {"Jan": 0, "Feb": 1, "Mar": 2, "Apr": 3, "May": 4, "June": 5, "Jul": 6,
                  "Aug": 7, "Sep": 8, "Oct": 9, "Nov": 10, "Dec": 11}
        user_type = {"New_Visitor": 0, "Returning_Visitor": 1, "Other": random.choice([0, 1])}
        weekend_revenue = {"TRUE": 1, "FALSE": 0}

        for user in users:

            local_data = []

            # iterate through each user data value and append it to local_data
            for i in range(len(user)):
                # change val to numerical before appending if needed
                if i == 10:
                    local_data.append(months[user[i]])
                elif i == 15:
                    local_data.append(user_type[user[i]])
                elif i == 16:
                    local_data.append(weekend_revenue[user[i]])

                # change val type before appending if needed
                elif i == 0 or i == 2 or i == 4 or i == 11 or i == 12 or i == 13 or i == 14:
                    local_data.append(int(user[i]))
                elif i == 1 or i == 3 or i == 5 or i == 6 or i == 7 or i == 8 or i == 9:
                    local_data.append(float(user[i]))

            # append the complete list of data for the current user
            users_data.append(local_data)

    # Make predictions
    predictions = model.predict(users_data)

    # Display predictions
    for i in range(len(predictions)):
        print(f"User {i}: {classification_dict[predictions[i]]}")


predict()

