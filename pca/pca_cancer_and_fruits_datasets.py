# PCA on cancer and fruits datasets

import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Breast cancer dataset
cancer = load_breast_cancer()
(X_cancer, y_cancer) = load_breast_cancer(return_X_y = True)

# fruits dataset
fruits = pd.read_table('readonly/fruit_data_with_colors.txt')
X_fruits = fruits[['mass','width','height', 'color_score']]
y_fruits = fruits[['fruit_label']] - 1

# PCA to find 2 main components of cancer dataset
X_normalized = StandardScaler().fit(X_cancer).transform(X_cancer)  
pca = PCA(n_components = 2).fit(X_normalized)
X_pca = pca.transform(X_normalized)
print(X_cancer.shape, X_pca.shape)

# PCA on fruits dataset
X_normalized = StandardScaler().fit(X_fruits).transform(X_fruits)
pca = PCA(n_components = 2).fit(X_normalized)
X_pca = pca.transform(X_normalized)



