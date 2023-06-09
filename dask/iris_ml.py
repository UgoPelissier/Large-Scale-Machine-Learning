import dask_ml
import joblib
import sklearn
from dask.distributed import Client, LocalCluster
import joblib
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import dask.dataframe as dd

import warnings
warnings.filterwarnings("ignore")

# load the data in a pandas dataframe
def load(data):
    iris_df = pd.read_csv(data)
    return iris_df

#prepocess
def preprocess(iris_df):
    # split the dataframe into labels and features
    X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = iris_df['variety']
    # convert the dataframe to dask array
    X = dd.from_pandas(X, npartitions=6).to_dask_array(lengths=True)
    y = dd.from_pandas(y, npartitions=2).to_dask_array(lengths=True)
    # split the data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test

# machine learning pipeline
def pipeline(data, model):
    # open the parallelization with dask
    with joblib.parallel_backend('dask'):
        X_train, X_test, y_train, y_test = preprocess(load(data))
        #choose a model
        if model == 'Decision_tree':
            clf = DecisionTreeClassifier()
        if model == 'Random_forest':
            clf = RandomForestClassifier()
        if model == 'Gradient_boost':
            clf = GradientBoostingClassifier()
        # train the model
        clf = clf.fit(X_train,y_train)
        # test the model
        y_pred = clf.predict(X_test)
        print('Accuracy - ',model,':', accuracy_score(np.asarray(y_test), np.asarray(y_pred)), sep='')

#open a dask client
# cluster = LocalCluster()
# client = Client(cluster)
#c = Client(n_workers=4)

pipeline('data/iris.csv', 'Decision_tree')
pipeline('data/iris.csv', 'Random_forest')
pipeline('data/iris.csv', 'Gradient_boost')