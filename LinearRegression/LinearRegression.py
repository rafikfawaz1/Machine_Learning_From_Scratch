
from random import weibullvariate
import numpy as np 

class LinearRegression:
    def __init__(self,lr=0.001,n_iters=1000) -> None:
        self.lr=lr
        self.n_iters=n_iters
        self.weights= None
        self.bias = None
    
    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias=0 
        
        for _ in range(self.n_iters):
            y_hat = np.dot(X,self.weights) + self.bias

            dw= (1/n_samples) * np.dot(X.T,(y_hat - y))
            db = (1/n_samples)* np.sum(y_hat - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db 
    def predict(self,X):
        y_hat= np.dot(X,self.weights) + self.bias 
        return y_hat

