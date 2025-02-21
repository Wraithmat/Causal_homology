from pyclassify.utils import distance, majority_vote 
from pyclassify.classifier import kNN
import pytest


def test_distance():
    v1=[0,1,2]
    v2=[1,3,4]
    v3=[1,1,1]
    
    with pytest.raises(TypeError):
        if distance(v1,v2)!=distance(v2,v1):
            raise Exception("The distance is not symmetric")
        elif distance(v1,v1)!=0:
            raise Exception("The distance among equal vectors is not zero")
        elif distance(v1,v2)+distance(v2,v3)<distance(v1,v3):
            raise Exception("The distance should follow the triangular inequality")
        elif distance([0,0,0],v3)!=3**0.5:
            raise Exception("The distance is computed in the wrong manner")
        else:
            raise TypeError()

def test_majority_vote():
    classes=[0,1,1,1,1,1,2,3,4]
    with pytest.raises(Exception):
        if majority_vote(classes)!=1:
            raise Exception(f"The class {majority_vote(classes)} has been computed, instead of 1")
    
def test_kNN():
    with pytest.raises(ValueError):
        kNN(backhand='pippo')
    with pytest.raises(TypeError):
        kNN(k='ciao')
    with pytest.raises(ValueError):
        kNN(k=-1)
