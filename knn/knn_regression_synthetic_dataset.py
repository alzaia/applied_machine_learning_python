# Example of knn regression using a simple synthetic dataset


import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# synthetic dataset for simple regression
X_R1, y_R1 = make_regression(n_samples = 100, n_features=1,
                            n_informative=1, bias = 150.0,
                            noise = 30, random_state=0)

# plot regression data                           
plt.figure()
plt.title('Sample regression problem with one input variable')
plt.scatter(X_R1, y_R1, marker= 'o', s=50)
plt.show()

# split train test
X_train, X_test, y_train, y_test = train_test_split(X_R1, y_R1, random_state = 0)

# perform knn regression with k = 5
knnreg = KNeighborsRegressor(n_neighbors = 5).fit(X_train, y_train)

# plot result and score
print(knnreg.predict(X_test))
print('R-squared test score: {:.3f}'.format(knnreg.score(X_test, y_test)))

