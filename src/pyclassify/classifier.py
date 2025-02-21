from .utils import *
import numpy as np

class kNN():
    def __init__(self,k=1,backhand='plain'):
        """
        inputs:
        k: number of nearest neighbors considered
        """
        if backhand!='numpy' and backhand!='plain' and backhand!='numba':
            raise ValueError('''Only 'numpy' and 'plain' are possible options''')
        if type(k)!=int:
            raise TypeError
        if k<=0:
            raise ValueError
        self.k=k
        if backhand=='plain':
            self.distance=distance
        if backhand=='numpy':
            self.distance=distance_numpy
        if backhand=='numba':
            self.distance=distance_numba

    @profile
    def _get_k_nearest_neighbors(self, X, y, x):
        """
        inputs:
        X: the dataset values
        y: dataset labels
        x: newpoint
        """
        distances=[]
        i=-1
        order=[]
        for datapoint in X:
            i+=1
            distances.append([self.distance(datapoint,x),i])
        distances=sorted(distances, key=lambda x: x[0])
        return [y[distances[p][1]] for p in range(min(self.k,len(distances)))]
    
    @profile
    def __call__(self,data,new_points):
        """
        data: tuple (matrix of points, labels)
        newpoints: matrix of points to classify
        returns
        classification: the labels of the different datapoints
        """
        classification=[]
        x=data[0]
        y=data[1]
        if self.distance==distance_numpy or self.distance==distance_numba:
            x=np.array(x)
            y=np.array(y)
            new_points=np.array(new_points)
        for point in new_points:
            classes=self._get_k_nearest_neighbors(x, y, point)
            classification.append(majority_vote(classes))
        return classification
