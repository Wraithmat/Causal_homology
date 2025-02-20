from .utils import distance, majority_vote

class kNN():
    def __init__(self,k=1):
        """
        inputs:
        k: number of nearest neighbors considered
        """
        self.k=k

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
            distances.append([distance(datapoint,x),i])
        distances=sorted(distances, key=lambda x: x[0])
        return [y[distances[p][1]] for p in range(min(self.k,len(distances)))]
        
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
        for point in new_points:
            classes=self._get_k_nearest_neighbors(x, y, point)
            classification.append(majority_vote(classes))
        return classification
