from random import random

import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

X,y = datasets.make_regression(n_samples=100,n_features=1,noise=20,random_state=4)
X_train , X_test,y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=1234)

# figure = plt.figure(figsize=(8,6))
# plt.scatter(X[:,0],y)
# plt.show()

# print (X_train.shape)
# print(y_train.shape)
from LinearRegression import LinearRegression

regressor = LinearRegression(lr=0.01)
regressor.fit(X_train,y_train)
predicted = regressor.predict(X_test)
# mse = MSE(y_test,predicted)
predicted_line = regressor.predict(X)
def MSE(y,y_hat):
    return np.mean((y-y_hat)**2)
    
mse = MSE(y_test,predicted)
print(mse)

def plot():
    plt.scatter(X_train,y_train)
    plt.scatter(X_test,y_test)
    plt.plot(X,predicted_line)
    plt.show()
plot()
