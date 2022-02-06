from collections import Counter
import numpy as np 
def euclidean_distance(x1,x2):
    return np.sqrt(np.sum(x1-x2)**2)
class KNN:
    def __init__(self,k=3):
        self.k=k

    def fit(self,X,y):
        self.X_train=X
        self.y_train=y


    def predict(self,X):
        predicted_labels = [self._predict(x)for x in X ]
        return np.array(predicted_labels)
    def _predict(self,x):
        # compute distances 
        distance = [euclidean_distance(x,X_train) for X_train in self.X_train ]
        # get k nearest samples 
        k_ind= np.argsort(distance)[:self.k]
        # get labels 
        k_nearest_labels = [self.y_train[i] for i in k_ind]
        # get most common class label 
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
