import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from tabulate import tabulate
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style 

train = False
verbose = False


data = pd.read_csv("../data/student/student-mat.csv", sep=";")
if verbose:
    print(data.head())

# Adjust which attributes you want to use
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
if verbose:
    print(data.head())

# Attribute prediction value
predict = "G3"

x = np.array(data.drop([predict], 1))
y = np.array(data[predict])

# Assign data train and test data sets 
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

if train:
    best = 0
    for _ in range(30):
        x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
        model = linear_model.LinearRegression()
        model.fit(x_train, y_train)

        accuracy = model.score(x_test, y_test)
        if verbose:
            print(accuracy) 
        if accuracy > best:
            best = accuracy
            with open("../models/studentmodel.pickle", "wb") as f:
                pickle.dump(model, f)

pickle_in = open("../models/studentmodel.pickle", "rb")
model = pickle.load(pickle_in)

if verbose:
    print("Coefficient: ", model.coef_)
    print("Intercept: ", model.intercept_)
 


predictions = model.predict(x_test)

table = []
for x in range(len(predictions)):
    table.append([predictions[x], x_test[x], y_test[x]])

print(tabulate(table, headers=["Guess", "Input", "Actual"], tablefmt='fancy_grid'))

p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()