
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.regression import r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve


# read mushroom dataset and take a subset

mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

X_subset = X_test2
y_subset = y_test2


# train decision tree classifier and find 5 most important features
model = DecisionTreeClassifier(random_state = 0).fit(X_train2, y_train2)
feature_names = []

for index, importance in enumerate(model.feature_importances_):
    feature_names.append([importance, X_train2.columns[index]])

feature_names.sort(reverse=True)
feature_names = np.array(feature_names)
feature_names = feature_names[:5,1]
feature_names = feature_names.tolist()


# train SVM with RBF kernel and use the validation curve module to get train/test scores for various gamma values
model = SVC(kernel='rbf', C=1, random_state=0)
gamma = np.logspace(-4,1,6)
train_scores, test_scores = validation_curve(model, X_subset, y_subset, param_name='gamma', param_range=gamma, scoring='accuracy')

mean_scores_train = train_scores.mean(axis=1)
mean_scores_test = test_scores.mean(axis=1)
    












