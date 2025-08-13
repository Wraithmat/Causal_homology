from pyclassify.utils import fast_smallest_ball, Cech_complex, homology_from_laplacian, homology_from_reduction, alpha_complex, collapsed_Cech_complex, collapsed_Cech_complex_with_sets, radius_selection, bayesian_multinomial_mode, max_principal_curvature
import copy
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

def test_cech_complex():
    # Test empty input or one dimensional one
    with pytest.raises(AssertionError):
        Cech_complex(np.array([[]]), 1, 2)
        Cech_complex(np.array([1,2,3]), 1, 2)

    # Test with a simple case
    points = np.array([[0,0],[1,1],[0,1],[1,1]])
    C = Cech_complex(points, 1.5, 2)
    assert len(C[0]) == 4  # All points are in the first complex
    assert len(C[1]) == 6  # All pairs of points are in the second complex
    C = collapsed_Cech_complex(points, 1.5, 2)
    assert len(C[0]) == 4  # All points are in the first complex
    C_ = collapsed_Cech_complex_with_sets(points, 1.5, 2)
    assert [len(C_[i]) == len(C[i]) for i in range(3)].count(True) == 3  # All complexes should be the same

def test_alpha_complex():
    # Test empty input or one dimensional one
    with pytest.raises(AssertionError):
        alpha_complex(np.array([[]]), 1, 2)
        alpha_complex(np.array([1,2,3]), 1, 2)

    # Test with a simple case
    points = np.array([[0,0],[1,1],[0,1],[1,1]])
    C = alpha_complex(points, 1.5, 2)
    assert len(C[0]) == 4  # All points are in the first complex

def test_homology():
    C={0:[[0],[1],[2],[3],[4],[5],[6],[7],[8]], 1:[[0,8],[0,1],[1,2],[0,4],[0,3],[0,2],[1,4],[1,5],[1,6],[1,7],[0,6],[2,5],[2,6],[2,7],[2,8],[3,4],[3,5],[3,6],[3,7],[3,8],[4,5],[4,7],[4,8],[5,6],[5,8],[6,7],[7,8]],2:[[0,1,4],[0,1,6],[1,2,5],[1,2,7],[0,2,6],[0,2,8],[1,6,7],[2,7,8],[0,3,8],[0,3,4],[1,4,5],[2,5,6],[3,4,7],[4,5,8],[3,5,6],[3,5,8],[4,7,8],[3,6,7]]}
    Op=homology_from_laplacian(C, max_k=25, sparse=False)
    C_p= copy.deepcopy(C)
    O_r=homology_from_reduction(C_p)
    assert Op == [1,1,0]
    assert O_r == [1,1]

def test_reach_estimations():
    points = np.array([[0,0],[1,1],[0,1]])
    r = radius_selection(points)
    assert r>0.45 and r<np.sqrt(2)

    prior = np.array([3,2,1])
    samples =[(1,2,3),(4,5,6),(7,8,9)]
    points = [(1,2,3),(4,5,6)]

    p, var = bayesian_multinomial_mode(prior, samples, points)
    assert np.argmax(p) == 0

    points = np.linspace(0,2*np.pi, 200)
    torus=np.vstack([np.cos(points[:]), np.sin(points[:])]).T

    C = max_principal_curvature(torus, NN=50, d=1, cross_val=(1.1,0.9))

    assert C>0.8 and C<1.2






