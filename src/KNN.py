import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
import numpy as np
from sklearn import linear_model, preprocessing
from tabulate import tabulate

data = pd.read_csv("../data/car/car.data")
print(data.head())

label_encoder = preprocessing.LabelEncoder()
buying = label_encoder.fit_transform(list(data["buying"]))
maintenance = label_encoder.fit_transform(list(data["maintenance"]))
door = label_encoder.fit_transform(list(data["door"]))
persons = label_encoder.fit_transform(list(data["persons"]))
lug_boot = label_encoder.fit_transform(list(data["lug_boot"]))
safety = label_encoder.fit_transform(list(data["safety"]))
cls = label_encoder.fit_transform(list(data["class"]))

predict = "class"

x = list(zip(buying, maintenance, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# play around with number of neighbors 
model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
accuracy = model.score(x_test, y_test)
print(accuracy)

predictions = model.predict(x_test)

names = ["unacc", "acc", "good", "vgood"]
table = []
for x in range(len(predictions)):
    table.append([names[predictions[x]], x_test[x], names[y_test[x]]])

print(tabulate(table, headers=["Guess", "Input", "Actual"], tablefmt='fancy_grid'))
