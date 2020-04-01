# Polynomial linear regression on synthetic dataset of sin function

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics.regression import r2_score

# create regression dataset with sin and added noise
np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

plt.figure()
plt.scatter(X_train, y_train, label='training data')
plt.scatter(X_test, y_test, label='test data')
plt.legend(loc=4);

# fit polynomial model with various degrees and test on values 0 to 10
values_to_predict = np.linspace(0,10,100)
i = [0, 1, 2, 3]
deg = [1, 3, 6, 9]
results = np.zeros((4,100))

for i, deg in zip(i, deg):
    poly = PolynomialFeatures(degree=deg)
    X_poly = poly.fit_transform(X_train.reshape(len(X_train),1))
    linreg = LinearRegression().fit(X_poly, y_train)
    results[i,:] = linreg.predict(poly.fit_transform(values_to_predict.reshape(100,1)))
    

# get R2 scores for train and test for models with degrees 0 to 9
i = np.arange(10)
deg = np.arange(10)
r2_train = np.zeros(10,)
r2_test = np.zeros(10,)

for i, deg in zip(i, deg):
    poly = PolynomialFeatures(degree=deg)
    X_poly_train = poly.fit_transform(X_train.reshape(len(X_train),1))
    X_poly_test = poly.fit_transform(X_test.reshape(len(X_test),1))
    linreg = LinearRegression().fit(X_poly_train, y_train)
    r2_train[i] = r2_score(y_train, linreg.predict(X_poly_train))
    r2_test[i] = r2_score(y_test, linreg.predict(X_poly_test))
        

# compare between polynomial linear model deg=12 and lasso polynomial model deg=12
poly = PolynomialFeatures(degree=12)
X_poly_train = poly.fit_transform(X_train.reshape(len(X_train),1))
X_poly_test = poly.fit_transform(X_test.reshape(len(X_test),1))

model1 = LinearRegression().fit(X_poly_train, y_train)
model2 = Lasso(alpha=0.01,max_iter=10000).fit(X_poly_train, y_train)

LinearRegression_R2_test_score = r2_score(y_test, model1.predict(X_poly_test))
Lasso_R2_test_score = r2_score(y_test, model2.predict(X_poly_test))








