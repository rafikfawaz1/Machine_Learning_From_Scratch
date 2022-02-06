import numpy as np 

class PCA:
    def __init__(self,n_components):
        self.n_components = n_components
        self.component= None
        self.mean= None
        
    def fit(self,X):
        # mean 
        self.mean= np.mean(X,axis=0)
        X = X - self.mean
        cov = np.cov(X.T)
        eigenvalues , eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T 
        # COV matrices 
        #eigen
        #sort 
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[idx]
        self.component = eigenvectors[0:self.n_components]

    def transform(self,X):
       X = X - self.mean 

       return np.dot(X,self.component.T)
