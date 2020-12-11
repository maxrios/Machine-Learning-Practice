# Support Vector Machine
# Keywords: Kernel 
import sklearn
from sklearn import datasets, svm, metrics
from sklearn.neighbors import KNeighborsClassifier


verbose = True

cancer = datasets.load_breast_cancer()
if verbose:
    print(cancer.feature_names)
    print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

if verbose:
    print(x_train, y_train)

classes = ['malignant', 'benign']

clf = svm.SVC(kernel="linear", C=2)
clf.fit(x_train, y_train)

y_prediction = clf.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_prediction)

if verbose:
    print("Accuracy:", accuracy)