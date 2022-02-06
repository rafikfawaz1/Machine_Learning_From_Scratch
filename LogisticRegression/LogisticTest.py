from random import random
from xml.sax.saxutils import prepare_input_source
# from statistics import linear_regression
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

bc=datasets.load_breast_cancer()
X,y =bc.data ,bc.target 
X_train , X_test,y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1234)

from LogisticRegression import LogisticRegression
regressor = LogisticRegression(lr=0.001,n_iters=1000)
regressor.fit(X_train,y_train)
prediction = regressor.predict(X_test)

def accuracy(ytest,yhat):
    accuracy = np.sum(ytest==yhat)/len(y_test)
    return accuracy
print ("accuray : ", accuracy (y_test,prediction))
