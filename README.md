# Customer intent predictor

This program aims to predict whether a customer intends to buy a product based on certain attributes defined by their behaviour during the visit to a website.
You can either use predict.py to get predictions for specific data, or you can tweak shopping.py and train new models and optionally save them.

## Requirements installation:

- You need to install python Itself
- The, run the following command on the project directory:

```bash
pip install scikit-learn
```

## Example usage scenarios:

### Run predictions on models pre-trained by me

The csv used for input during predictions follows the same format as the training dataset, however it should contain neither the header row (with the titles) nor the last column (Revenue)

- Second parameter: model
- Third parameter: csv file with each row representing each user to make predictions on

```python
python predict.py models/model1.sav try_user_data/users1.csv
```
(don't forget file extensions)

### Tweak shopping.py and train models yourself

- Second parameter: csv dataset to train on
- Third parameter (optional): path to save the model

```python
python shopping.py shopping.csv models/model2.sav
```