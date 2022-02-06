import numpy as np 


class LogisticRegression:
    def __init__(self,lr,n_iters):
        self.lr=lr 
        self.n_iters = n_iters
        self.weights=None
        self.bias = None 

    
    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias= 0
        print(n_samples)

        for _ in range(self.n_iters):
            linear_model= np.dot(X,self.weights) + self.bias
            y_hat = self._sigmoid(linear_model)

            dw = (1/n_samples) * np.dot(X.T,(y_hat-y))
            db = (1/n_samples)* np.sum(y_hat-y) 
            self.weights -= self.lr - dw 
            self.bias -= self.bias-db 

    def _sigmoid(self,x):
        return 1/(1+np.exp(-x))
    

    def predict(self,X):
        linear_model= np.dot(X,self.weights) + self.bias
        y_hat = self._sigmoid(linear_model)
        y_class = [1 if i > 0.5 else 0 for i in y_hat]
        return y_class

