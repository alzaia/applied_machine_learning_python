

# Imports
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

cancer = load_breast_cancer()
print(cancer.DESCR) # Print the data set description
cancer.keys()

def cancer_to_dataframe():
    
    # data: shape (569,31) with 30 first columns = features and last column = target
    data = np.column_stack((cancer.data, cancer.target))
    # names: 30 features names + target at the end
    column_names = np.append(cancer.feature_names, 'target')
    # index from 0 to 569
    index = pd.RangeIndex(start=0, stop=569, step=1)
    # create DataFrame object
    cancer_data = pd.DataFrame(data=data, index=index, columns=column_names)
    
    return cancer_data

# Get cancer distribution (malignant vs benign)
    
    # get DataFrame object
    data_table = cancer_to_dataframe()
    # create index
    index = ['malignant', 'benign']
    # count number of malignant and benign cases
    nbr_mal = np.count_nonzero(data_table['target'] == 0.0)
    nbr_ben = len(data_table['target']) - nbr_mal
    # combine both counts
    counts_array = [nbr_mal, nbr_ben]
    # combine everything in a Series object
    target = pd.Series(data=counts_array,index=index)

def split_into_X_y():
    
    cancerdf = cancer_to_dataframe()

    X = cancerdf.drop('target', axis=1)
    y = cancerdf.get('target')
    
    return X, y

def split_into_train_test():

    X, y = split_into_X_y()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    return X_train, X_test, y_train, y_test


def train_knn():
    X_train, X_test, y_train, y_test = split_into_train_test()
    
    knn = KNeighborsClassifier(n_neighbors = 1)
    knn.fit(X_train, y_train)
    
    return knn

# Predict class of a mean cancer tumor
    cancerdf = cancer_to_dataframe()
    means = cancerdf.mean()[:-1].values.reshape(1, -1)
    knn = train_knn()
    
    cancer_prediction = knn.predict(means)
    print(cancer_prediction)
    
# Predict test
    X_train, X_test, y_train, y_test = split_into_train_test()
    knn = train_knn()

    cancer_prediction = knn.predict(X_test)
    print(cancer_prediction)
    
# Check accuracy of test data
    X_train, X_test, y_train, y_test = split_into_train_test()
    knn = train_knn()
    
    test_score = knn.score(X_test, y_test)
    print(test_score)
    
