# Stock price prediction

from numpy.__config__ import show
import quandl 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

quandl.ApiConfig.api_key = "LxL4ZxWP8skRHemZgHss"

verbose = True
show_results = True

# Retrieving stock data
df = quandl.get("CHRIS/FB") 
if verbose:
    print(df.tail())

# Filtering data
df = df[['Adj. Close']]
if verbose:
    print(df.head())

# Prediction
# Variable for predicting n days in future
forecast_out = 1
# New column shifted n units up
df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)
if verbose:
    print(df.head())

# Independent data set
X = np.array(df.drop(['Prediction'], 1))
# Remove last n rows
X = X[:-forecast_out]
if verbose:
    print(X)

# Dependent data set 
y = np.array(df['Prediction'])
y = y[:-forecast_out]
if verbose:
    print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# SVM (Regressor)
svr_rbf = SVR(kernel='rbf', C=1e3, gamma= 0.1)
svr_rbf.fit(x_train, y_train)
svm_confidence = svr_rbf.score(x_test, y_test)
if show_results:
    print("svm confidence:", svm_confidence)

# Linear Regression Model
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_confidence = lr.score(x_test, y_test)
if show_results:
    print("lr confidence:", lr_confidence)


x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast_out:]
if verbose:
    print(x_forecast)

lr_prediction = lr.predict(x_forecast)
if show_results:
    print('Linear Regression:', lr_prediction)

svm_prediction = svr_rbf.predict(x_forecast)
if show_results:
    print('SVM:', svm_prediction)

