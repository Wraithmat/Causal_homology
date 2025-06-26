from pyclassify.utils import fast_smallest_ball, Cech_complex
import numpy as np
from scipy.special import binom
import pytest

def test_fast_smallest_ball():
    
    with pytest.raises(AssertionError):
        #Test empty input or one dimensional one
        fast_smallest_ball(np.array([[]]))
        fast_smallest_ball(np.array([1,2,3]))
    
    # check that the correct radius is found
    assert fast_smallest_ball(points=np.array([[0,0],[1,1],[0,1],[1,1]]), return_radius=True)==np.sqrt(2)/2
    
    # check that indeed the distance between the points in the border and the center is always equal
    points = np.array([[0,0],[2,2],[0,2],[1.2,1.345]])
    c,T = fast_smallest_ball(points, return_radius=False)
    assert np.allclose(np.sum((points[T] - c)**2, axis=1), np.sum((points[T[0]] - c)**2))

    # check that when you have points very close to each other, then the number of complexes which are built is given by the binomial coefficient
    points = np.random.uniform(0,1,(15,2))
    C = Cech_complex(points, 10, 3)
    assert np.allclose([binom(15,i+1) for i in range(4)],[len(C[i]) for i in range(4)])

    
# You can check that the the matrix laplacian has only 0,1,-1 entries (off diagonal?)

