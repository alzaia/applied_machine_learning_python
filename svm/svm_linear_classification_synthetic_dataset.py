

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.svm import LinearSVC

# synthetic dataset for classification (binary) 
X_C2, y_C2 = make_classification(n_samples = 100, n_features=2,
                                n_redundant=0, n_informative=2,
                                n_clusters_per_class=1, flip_y = 0.1,
                                class_sep = 0.5, random_state=0)

plt.figure()
plt.title('Sample binary classification problem with two informative features')
plt.scatter(X_C2[:, 0], X_C2[:, 1], c=y_C2, marker= 'o', s=50, cmap=cmap_bold)
plt.show()

# train test split
X_train, X_test, y_train, y_test = train_test_split(X_C2, y_C2, random_state = 0)

# train model
this_C = 1.0
clf = SVC(kernel = 'linear', C=this_C).fit(X_train, y_train)

# test with different C values
for this_C in [0.00001, 100]:
    clf = LinearSVC(C=this_C).fit(X_train, y_train)






