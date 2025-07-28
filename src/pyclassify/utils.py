import numpy as np
from scipy.spatial import distance_matrix, Delaunay
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import eigsh, svds
from scipy.linalg import eigh
from line_profiler import profile
from joblib import Parallel, delayed
import copy
from scipy.stats import gaussian_kde, beta
from scipy.signal import find_peaks
#from sklearn.cluster import AgglomerativeClustering #################!!!!!!!!! not yet in the requirements
from scipy.optimize import minimize, shgo
from tqdm import tqdm
from numba import njit

"""
We compute the homology group in different steps:

1. build the simplicial complexes needed for the identification of loops
    1.1. using the smallest ball algorithm to build Čech
    1.2. building the Vietoris-Rips complex
    (possibly starting from a distance matrix, as it is done for k-means clustering with the multi-dimensional scaling ideas)

2. analyse the homology group
    2.1. with the straightforward @profile
definition
    2.2. using the Laplacian operator
    2.3. using persistent homology

We test the algorithm in optimal theoretical setups and check:
    1. that we can retrieve the correct homology with the correct probability (as predicted in Niyogi, P., Smale, S. & Weinberger, S. "Finding the Homology of Submanifolds with High Confidence from Random Samples", proposition 3.2)
    2. all for different kind of shapes (e.g. two disconnected rings, Bolza surface, torus)
    (possibly we could compute an estimate of the reach number and check if numerically it is possible to use this information to understand if we are in proper regime)
"""

def slice_dok_0(mat, row_indices, col_indices):
    """Slice a dok_matrix and return another dok_matrix."""
    result = dok_matrix((len(row_indices), len(col_indices)), dtype=mat.dtype)
    
    row_map = {i: idx for idx, i in enumerate(row_indices)}
    col_map = {j: idx for idx, j in enumerate(col_indices)}
    
    for (i, j), val in mat.items():
        if i in row_map and j in col_map:
            result[row_map[i], col_map[j]] = val
    
    return result

def slice_dok(dok, row_indices, col_indices):
    """
    Slice a dok_matrix by selecting specified rows and columns.
    This version uses CSR and CSC for efficiency.
    
    Parameters:
        dok (dok_matrix): Input sparse matrix.
        row_indices (array-like): List or array of row indices to keep.
        col_indices (array-like): List or array of column indices to keep.
        
    Returns:
        dok_matrix: Sliced result as DOK matrix.
    """
    csr = dok.tocsr()

    sliced_csr = csr[row_indices, :]

    csc = sliced_csr.tocsc()

    sliced_csc = csc[:, col_indices]

    result = sliced_csc.todok()

    return result

def slice_dok_columns_0(mat, col_indices):
    """Slice only columns of a dok_matrix and return a new dok_matrix."""
    n_rows = mat.shape[0]
    result = dok_matrix((n_rows, len(col_indices)), dtype=mat.dtype)
    
    col_map = {j: new_j for new_j, j in enumerate(col_indices)}
    
    for (i, j), val in mat.items():
        if j in col_map:
            result[i, col_map[j]] = val
    
    return result

def slice_dok_columns(dok, keep_cols):
    """Remove specified columns from a dok_matrix using CSC format."""
    csc = dok.tocsc()
    
    sliced_csc = csc[:, keep_cols]

    return sliced_csc.todok()

def slice_dok_rows(dok, keep_rows):
    """Remove specified rows from a dok_matrix using CSR format."""
    csr = dok.tocsr()
    
    sliced_csr = csr[keep_rows, :]

    return sliced_csr.todok()

def heuristic_ordering(points):
    """In order to optimize the number of elementary collapses, we can use an heuristic rule: we order the points by the distance from the center of mass"""

    center = np.mean(points, axis=0)
    distances = distance_matrix(center.reshape(1,-1), points)[0]
    order = np.argsort(-distances)

    return points[order]

@profile
def _project_affine_space(p, points, eps=1e-9):
    """
    Parameters:
        p : array-like; a point in D dimensions which must be projected in the affine space of points
        points : array-like; the not orthogonal basis of the affine space
        Returns:
        cc : array-like; the projection on the affine
    Returns:
        points: the projection of p on the affine space
        coeff: the full list of coefficients of the projection
        check (bool): True if the projection and the starting point coincide (i.e. if the point was already in the affine space)
    """
    if len(points) == 0:
        raise ValueError("T should not be empty")
    if len(points) == 1:
        return points.reshape(-1) , [1.0], 0
    else:
        # fix the first point as pivotal point
        A = (points[1:]-points[0]).T
        Q, R = np.linalg.qr(A, mode='complete')
        R = R[:A.shape[1]]
        y = Q.T@(p-points[0])
        y = y[:A.shape[1]]
        coeff = np.linalg.solve(R, y)

        matrix = A @ coeff +points[0]
        check = max(np.abs(matrix-p))<eps
        # Oss: if A@cc==c-points[T[0]] then the center was already on the affine space
        # the coefficients are given by cc[i] for all elements of T[i+1], T[0] coefficient is 1-sum(cc)x
        
        sum_coeff = np.sum(coeff)
        coeff_list = coeff.tolist()
        return matrix.ravel(), [1 - sum_coeff] + coeff_list, check
        #return (matrix).reshape(-1), [1-np.sum(coeff)]+coeff.tolist(), check
    
@profile
def _walking_step(points, c, cc, T, eps=0):
    """
    This function is used to find the correct displacement of the center of the ball
    """

    v = cc-c

    tmax = 1
    i_star = -1
    for i in range(len(points)):
        if i in T:
            continue
        if np.dot(points[i]-c,v) >= np.dot(v,v):
            continue # p is always inside the ball
        #den = np.sum(-2*(points[i]*v)) - np.sum(-2*(points[T[0]]*v))
        den = -2 * np.dot(points[i] - points[T[0]], v)
        displ_0 = points[T[0]]-c
        displ_1 = points[i]-c
        #num = np.sum((points[T[0]]-c)**2) - np.sum((points[i]-c)**2)
        num = np.dot(displ_0,displ_0) - np.dot(displ_1, displ_1)
        t = num/den if den!=0 else 0 #If the denominator is 0, then there is a point that is already on the border that was not considered
        if t>0 and t < tmax+eps:
            tmax = t
            i_star = i
            #tmax = min(tmax, (t>0)*t+(t<0)*tmax)
        if t == 0:
            tmax = 0
            i_star = i
            break
            
    # Oss: it could be updated so that several i_star which give close tmax are kept, then the choice should go on i_star which is farthest from the affine space
    return tmax, i_star

@profile
def fast_smallest_ball(points, distances=None, maxiter=1000, return_radius=True):
    """
    Given a set of N points in D dimensions it finds the radius of the smallest ball which enclose them all.

    The implementation is based on the algorithm from the paper "Fast Smallest-Enclosing-Ball Computation in High Dimensions"
    by Kaspar Fischer, Bernd Gärtner, and Martin Kutz (Proc. 11th European Symposium on Algorithms (ESA), p. 630-641, 2003)

    Parameters:
        points : array-like; an NxD array with the position of the datapoints
        distances : array-like; precomputed distances between all points
        maxiter: in; maximum number of iterations allowed to find the smallest enclosing ball
        return_radius: bool; if True, only the radisus is returned

    Returns:
        r : float; radius of the smallest enclosing ball
        or if return_radius is False:
        c: array-like; the center of the smallest enclosing ball
        T: list; the indices of the points in the border of the SEB

    References:
        https://github.com/hbf/miniball/blob/master/cpp/main/Seb-inl.h
    """

    assert len(points.shape) == 2, "Points should be a 2D array of shape (N,D)"
    assert points.shape[1] > 0, "Points should not be empty"
    # Notice: we are mostly focusing on the case in which the total number of points is smaller than the number of dimensions

    # Initialize the center of the sphere on one point, T is then given by the index of the furthest point from there
    # In this way we start with a ball whom radius is at most twice as large as the radius of SEB
    c = points[0]
    if distances is None:
        distances = distance_matrix(points[0].reshape([1,-1]), points[1:])
        T = [np.argmax(distances[0])+1]
    else:
        T = [np.argmax(distances[0])] # T is the list of indices of the points in the border

    # Check c in aff(T)
    # c in aff(T), decompose and select negative coefficient
    # move c towards aff(T)
    # check if a point hits the boundary before reaching aff(T)
    # if a point stopped us, then T must be updated

    for _ in range(maxiter):
        cc, coeff, check = _project_affine_space(c, points[T])
        if check:
            if np.all(np.array(coeff) >= 0):
                #if np.any(np.array(coeff) > 0.5+1e-3): # we add a small epsilon to avoid machine resolution errors
                #    raise ValueError("Lemma 2 (Fischer et al.) is not satisfied")
                #else:
                    return np.linalg.norm(c - points[T[0]]) if  return_radius else (c,T)
                    #return np.sum((c-points[T[0]])**2)**0.5 if return_radius else (c,T)
            T.pop(np.argmin(coeff))
        else:
            tmax, i_star = _walking_step(points, c, cc, T)
            if i_star != -1:
                T.append(i_star)
            c = cc * tmax + (1-tmax)*c 

    raise ValueError("The algorithm di not converge within the maximum number of iterations.")

@profile
def _Vietoris_Rips_complex(points, two_epsilon_matrix, simplices=None):
    """
    A subroutine used to build the next order of simplices in Vietoris Rips complex and Cech complex
    """

    if simplices is None:
        return [[i] for i in range(len(points))]
    
    if len(simplices[0])==1:
        links = np.nonzero(np.triu(two_epsilon_matrix))
        return [[links[0][i],links[1][i]] for i in range(len(links[0]))]
    
    new_simplices = []
    for complex in simplices:
        # Check that there exist a point which is within 2-epsilon neighbourhood of all points in the complex analysed
        # take a pivotal node and check if any of its neighbours is connected to all the others
        for connected_node in np.nonzero(two_epsilon_matrix[complex[0]])[0]:            
            if np.sum(two_epsilon_matrix[connected_node, complex]) == len(complex):
                # If the node is connected to all the others, then we can add it to the complex
                new_complex = list(np.sort(complex + [connected_node]))
                if new_complex not in new_simplices:
                    new_simplices.append(new_complex)
    return new_simplices

@profile
def Cech_complex(points, epsilon, max_complex_dimension=2):
    """
    Given a set of point and a radius epsilon, it builds the Cech complex.
    
    Parameters:
        points: array-like; an NxD array with the position of the datapoints
        epsilon: float; the radius of the balls used to build the complex
    Returns:
        simplices: dict; a dictionary containing the simplices of the Cech complex
    """
    assert epsilon>0, "Epsilon should be a positive number"
    assert max_complex_dimension >= 0, "Max complex dimension should be a non-negative integer"
    assert len(points.shape) == 2, "Points should be a 2D array of shape (N,D)"
    assert points.shape[0] > 0, "Points should not be empty"

    # We introduce a distance matrix that allows to fasten the computations
    two_epsilon_matrix = distance_matrix(points, points) + 3*epsilon*np.eye(len(points))
    two_epsilon_matrix = two_epsilon_matrix < 2*epsilon

    simplices = {}

    for i in range(max_complex_dimension + 1):
        Viet_Rips_complex = _Vietoris_Rips_complex(points, two_epsilon_matrix, simplices[i-1] if i > 0 else None)
        Cech = []
        if i < 2:
            Cech = Viet_Rips_complex  
        else:
            for simplex in Viet_Rips_complex:
                r = fast_smallest_ball(points[simplex])
                if r<epsilon:
                    Cech.append(simplex)
        simplices[i] = Cech
        if Cech == []:
            for j in range(i, max_complex_dimension + 1):
                simplices[j] = []
            return simplices
    
    return simplices

@profile
def collapsed_Cech_complex(points, epsilon, max_complex_dimension=2):
    """
    Given a set of point and a radius epsilon, it builds the Cech complex while using the elementary simplicial collapse method to reduce it.
    Notice that this is not necessarily the minimal Cech complex that can be obtained by elementary collapse.
    
    Parameters:
        points: array-like; an NxD array with the position of the datapoints
        epsilon: float; the radius of the balls used to build the complex
    Returns:
        simplices: dict; a dictionary containing the collapsed simplices of the Cech complex
    """
    assert epsilon>0, "Epsilon should be a positive number"
    assert max_complex_dimension >= 0, "Max complex dimension should be a non-negative integer"
    assert len(points.shape) == 2, "Points should be a 2D array of shape (N,D)"
    assert points.shape[0] > 0, "Points should not be empty"

    
    simplices = {0:[[i] for i in range(len(points))]}


    two_epsilon_matrix = distance_matrix(points, points) + 3*epsilon*np.diag(np.ones(len(points)))
    two_epsilon_matrix = two_epsilon_matrix < 2*epsilon

    #row_links = lambda matrix: [np.nonzero(i) for i in matrix]

    #links = row_links(two_epsilon_matrix) #notice: in this version you can not use the upper triangular part only, you need to check the total number of links per row

    simplices[1]=[]
    for i in range(len(points)):

        # Version that does not compute the whole matrix of distances: notice, it compute twice all the distances
        #links = distance_matrix(points[i].reshape([1,-1]), points)
        #links[i]+=3*epsilon
        #links = np.nonzero(links[0] < 2*epsilon)[0] ## maybe specify that you focus on last part of the array
        
        links = np.nonzero(two_epsilon_matrix[i])

        if len(links[0])==1:
            two_epsilon_matrix[i] = 0
            two_epsilon_matrix[:,i] = 0
            simplices[0].remove([i])
            if [links[0], i] in simplices[1]:
                simplices[1].remove([links[0], i])
        elif len(links[0])>1:
            for j in links[0]:
                if j>i:
                    simplices[1].append([i,j])
    if simplices[1]==[]:
        for j in range(1, max_complex_dimension + 1):
            simplices[j] = []
        return simplices
    
    for i in range(2,max_complex_dimension+1):
        simplices[i]=[]
        for complex in np.copy(simplices[i-1]):
            # Check that there exist a point which is within 2-epsilon neighbourhood of all points in the complex analysed
            # take a pivotal node and check if any of its neighbours is connected to all the others
            Cech = []
            for connected_node in np.nonzero(two_epsilon_matrix[complex[0]])[0]:
                simplex = list(np.sort(list(complex) + [connected_node]))
                boundary = [simplex[:k]+simplex[k+1:] for k in range(i+1)]   
                #if np.sum([boundary[j] in simplices[i-1] for j in range(i+1)]) == i+1:
                if all(boundary[j] in simplices[i-1] for j in range(i+1)):
                    # If the complex with the new node has all the boundaries, then we check if it is a Cech simplex
                    r = fast_smallest_ball(points[simplex])
                    if r<epsilon:
                        Cech.append(simplex)
            if len(Cech)==1:
                simplices[i-1].remove(list(complex))
                if Cech[0] in simplices[i]:
                    simplices[i].remove(Cech[0])
            elif len(Cech)>1:
                for new_complex in Cech:
                    if new_complex not in simplices[i]:
                        simplices[i].append(new_complex)
        if simplices[i]==[]:
            for j in range(i, max_complex_dimension + 1):
                simplices[j] = []
            return simplices
            
    return simplices

@profile
def collapsed_Cech_complex_with_sets(points, epsilon, max_complex_dimension=2):
    """
    Given a set of point and a radius epsilon, it builds the Cech complex while using the elementary simplicial collapse method to reduce it.
    Notice that this is not necessarily the minimal Cech complex that can be obtained by elementary collapse.
    
    Parameters:
        points: array-like; an NxD array with the position of the datapoints
        epsilon: float; the radius of the balls used to build the complex
    Returns:
        simplices: dict; a dictionary containing the collapsed simplices of the Cech complex
    """
    assert epsilon>0, "Epsilon should be a positive number"
    assert max_complex_dimension >= 0, "Max complex dimension should be a non-negative integer"
    assert len(points.shape) == 2, "Points should be a 2D array of shape (N,D)"
    assert points.shape[0] > 0, "Points should not be empty"

    
    simplices = {0:set(tuple([i]) for i in range(len(points)))}


    two_epsilon_matrix = distance_matrix(points, points) + 3*epsilon*np.diag(np.ones(len(points)))
    two_epsilon_matrix = two_epsilon_matrix < 2*epsilon

    #row_links = lambda matrix: [np.nonzero(i) for i in matrix]

    #links = row_links(two_epsilon_matrix) #notice: in this version you can not use the upper triangular part only, you need to check the total number of links per row

    simplices[1]=set([])
    for i in range(len(points)):

        # Version that does not compute the whole matrix of distances: notice, it compute twice all the distances
        #links = distance_matrix(points[i].reshape([1,-1]), points)
        #links[i]+=3*epsilon
        #links = np.nonzero(links[0] < 2*epsilon)[0] ## maybe specify that you focus on last part of the array
        
        links = np.nonzero(two_epsilon_matrix[i])

        if len(links[0])==1:
            two_epsilon_matrix[i] = 0
            two_epsilon_matrix[:,i] = 0
            simplices[0].remove(tuple([i]))
            if tuple(list(links[0])+[i]) in simplices[1]:
                simplices[1].remove(tuple(list(links[0])+[i]))
        elif len(links[0])>1:
            for j in links[0]:
                if j>i:
                    simplices[1].add(tuple([i,j]))
    if simplices[1]==[]:
        for j in range(1, max_complex_dimension + 1):
            simplices[j] = set([])
        return simplices
    
    for i in range(2,max_complex_dimension+1):
        simplices[i]=set([])
        for complex in copy.copy(simplices[i-1]):
            # Check that there exist a point which is within 2-epsilon neighbourhood of all points in the complex analysed
            # take a pivotal node and check if any of its neighbours is connected to all the others
            Cech = set()
            for connected_node in np.nonzero(two_epsilon_matrix[complex[0]])[0]:
                simplex = list(sorted(list(complex) + [connected_node]))
                if all(tuple(simplex[:k] + simplex[k+1:]) in simplices[i-1] for k in range(i+1)):
                    # If the complex with the new node has all the boundaries, then we check if it is a Cech simplex
                    r = fast_smallest_ball(points[simplex])
                    if r<epsilon:
                        Cech.add(tuple(simplex))
            if len(Cech)==1:
                simplices[i-1].remove(complex)
                for el in Cech:
                    if el in simplices[i]: 
                        simplices[i].remove(el)
            elif len(Cech)>1:
                for new_complex in Cech:
                    if new_complex not in simplices[i]:
                        simplices[i].add(new_complex)
        if simplices[i]==[]:
            for j in range(i, max_complex_dimension + 1):
                simplices[j] = set([])
            return {list(simplices[i]) for i in simplices}
            
    return {i:list(simplices[i]) for i in simplices}

@profile
def alpha_complex(points, epsilon, max_complex_dimension=2):
    """
    By computing the intersection between a Cech complex and a Delaunay triangulation, it is possible to build a complex which contains a number of simplices linear in the number of points.
    Parameters:
        points: array-like; an NxD array with the position of the datapoints
        epsilon: float; the radius of the balls used to build the complex
    Returns:
        simplices: dict; a dictionary containing the collapsed simplices of the Cech complex
    """
    assert epsilon>0, "Epsilon should be a positive number"
    assert max_complex_dimension >= 0, "Max complex dimension should be a non-negative integer"
    assert len(points.shape) == 2, "Points should be a 2D array of shape (N,D)"
    assert points.shape[0] > 0, "Points should not be empty"

    delaunay_triangulation = Delaunay(points)
    index_del, neigh_del = delaunay_triangulation.vertex_neighbor_vertices

    max_complex_dimension = min(max_complex_dimension, points.shape[1]+1)

    simplices = {0:set(tuple([i]) for i in range(len(points)))}

    two_epsilon_matrix = distance_matrix(points, points) + 3*epsilon*np.diag(np.ones(len(points)))
    two_epsilon_matrix = two_epsilon_matrix < 2*epsilon

    simplices[1]=set([])
    for i in range(len(points)):   
        cech_links = np.nonzero(two_epsilon_matrix[i])
        delaunay_links = neigh_del[index_del[i]:index_del[i+1]]
        links=np.intersect1d(cech_links[0], delaunay_links, assume_unique=True)
        mask = np.zeros(len(points), dtype=bool)
        mask[links] = True
        two_epsilon_matrix[i] = mask
        two_epsilon_matrix[:,i] = mask
        if len(links)==1:
            two_epsilon_matrix[i] = 0
            two_epsilon_matrix[:,i] = 0
            simplices[0].remove(tuple([i]))
            if tuple([links[0], i]) in simplices[1]:
                simplices[1].remove(tuple([links[0], i]))
        elif len(links)>1:
            for j in links:
                if j>i:
                    simplices[1].add(tuple([i,j]))
    if simplices[1]==[]:
        for j in range(1, max_complex_dimension + 1):
            simplices[j] = set([])
        return simplices
    
    for i in range(2,max_complex_dimension+1):
        simplices[i]=set([])
        for complex in copy.copy(simplices[i-1]):
            # Check that there exist a point which is within 2-epsilon neighbourhood of all points in the complex analysed
            # take a pivotal node and check if any of its neighbours is connected to all the others
            Cech = set()
            for connected_node in np.nonzero(two_epsilon_matrix[complex[0]])[0]:
                simplex = list(sorted(list(complex) + [connected_node]))
                if all(tuple(simplex[:k] + simplex[k+1:]) in simplices[i-1] for k in range(i+1)):
                    # If the complex with the new node has all the boundaries,
                    if any(all(el in del_simplex for el in simplex) for del_simplex in delaunay_triangulation.simplices):
                        # we check if it is a possible Delaunay simplex 
                        # then we check if it is a Cech simplex
                        r = fast_smallest_ball(points[simplex])
                        if r<epsilon:
                            Cech.add(tuple(simplex))
            if len(Cech)==1:
                simplices[i-1].remove(complex)
                for el in Cech:
                    if el in simplices[i]: 
                        simplices[i].remove(el)
            elif len(Cech)>1:
                for new_complex in Cech:
                    if new_complex not in simplices[i]:
                        simplices[i].add(new_complex)
        if simplices[i]==[]:
            for j in range(i, max_complex_dimension + 1):
                simplices[j] = set([])
            return {list(simplices[i]) for i in simplices}
            
    return {i:list(simplices[i]) for i in simplices}

@profile
def Vietoris_Rips_complex(points, epsilon, max_complex_dimension=2):
    """
    Given a set of point and a radius epsilon, it builds the Vietoris Rips complex up to order max_complex_dimension.
    
    Parameters:
        points: array-like; an NxD array with the position of the datapoints
        epsilon: float; the radius of the balls used to build the complex
    Returns:
        simplices: dict; a dictionary containing the simplices of the Vietoris Rips complex
    """
    assert epsilon>0, "Epsilon should be a positive number"
    assert max_complex_dimension >= 0, "Max complex dimension should be a non-negative integer"
    assert len(points.shape) == 2, "Points should be a 2D array of shape (N,D)"
    assert points.shape[0] > 0, "Points should not be empty"

    # We introduce a distance matrix that allows to fasten the computations
    two_epsilon_matrix = distance_matrix(points, points) + 3*epsilon*np.diag(np.ones(len(points)))
    two_epsilon_matrix = two_epsilon_matrix < 2*epsilon

    simplices = {}

    for i in range(max_complex_dimension + 1):
        simplices[i] = _Vietoris_Rips_complex(points, two_epsilon_matrix, simplices[i-1] if i > 0 else None)
        if simplices[i] == []:
            for j in range(i, max_complex_dimension + 1):
                simplices[j] = []
                return simplices
    
    return simplices

@profile
def pair_reduction(E, B, i, index_a, index_b):
    """
    Parameters:
        E: dict; the collection of all the complexes (which are lists)
        B: dict; the collection of all the boundary matrices (which are sparse matrices)
        i: int; index of the complex to which b belongs
        a,b: lists; the generators that can be reduced
    Returns:
        E: dict; reduced collection of complexes, done in place
        B: dict; reduced collection of boundary matrices, done in place
    """
    
    #assert i+1 == len(b), f"The dimension of the generator b {b} should be equal to i+1 {i+1}"
    assert (i in E.keys()) and (i-1 in E.keys()) , f"There should be a complex indexed i {i} and {i-1}, instead we have {E.keys()}"
    assert ((i in B.keys()) and (i-1 in E.keys())) if i>1 else i in B.keys(), f"There should be a boundary matrix indexed i {i}, instead we have {B.keys()}"
    #assert (len(a)==len(b)-1) and (a!=[]), f'a {a} should be smaller than b {b} and not empty'
    #assert (b in E[i]) and (a in E[i-1]), f'a and b must be inside the complex lists'

    #####print('Removing', a, b)
    #index_a = E[i-1].index(a)
    #index_b = E[i].index(b)

    if i+1 in E.keys():
        mask = np.ones(B[i+1].shape[0], dtype=bool)
        mask[index_b] = 0
        keep_rows = np.flatnonzero(mask)
        B[i+1] = slice_dok_rows(B[i+1], keep_rows)
        #B[i+1]=B[i+1][mask]

    #columns_to_change = B[i][index_a].nonzero()[0].reshape(1,-1)
    #rows_to_change = B[i][:,index_b].nonzero()[0].reshape(1,-1)
    #print(rows_to_change, columns_to_change)
    #####print('changing', rows_to_change, columns_to_change)
    #####print(B[i][index_a],B[i][index_a].nonzero(), B[i][:,index_b], B[i][:,index_b].nonzero())
    #####print(B[i].todense())
    #####print(B[i-1].todense())

    ##grid = np.meshgrid(rows_to_change, columns_to_change)

    bdba = B[i][index_a,index_b]
    row_ = B[i][index_a, :]
    col_ = B[i][:, index_b]
    columns_to_change = row_.nonzero()[1]
    rows_to_change = col_.nonzero()[0]
    

    #print('***********---------------', row_.shape, col_.shape)

    ##B[i][grid[0].T,grid[1].T] -= bdba * B[i][index_a*np.ones(grid[1].T.shape), grid[1].T] * B[i][grid[0].T, index_b*np.ones(grid[1].T.shape)]
    #M = B[i][rows_to_change,columns_to_change.T].toarray()
    #B[i][rows_to_change,columns_to_change.T] = M - bdba * B[i][index_a*np.ones(rows_to_change.shape), columns_to_change.T].multiply(B[i][rows_to_change, index_b*np.ones(columns_to_change.shape).T])

    
    columns_to_change = list(columns_to_change)
    columns_to_change.remove(index_b)
    # Update elements individually due to sparse matrix indexing limitations
    for row in rows_to_change:
        for col in columns_to_change:
            #print('before:      ', row, col, '---', B[i][row, col])
            #print('#',bdba, B[i][index_a, col], B[i][row, index_b])
            B[i][row, col] -= bdba * row_[0, col] * col_[row, 0]
            #print('After:         ',B[i][row,col])

    E[i].pop(index_b)
    E[i-1].pop(index_a)


    mask = np.ones(B[i].shape[1], dtype=bool)
    mask[index_b] = 0
    keep_cols = np.flatnonzero(mask)

    mask_row = np.ones(B[i].shape[0], dtype=bool)
    mask_row[index_a] = 0
    keep_rows = np.flatnonzero(mask_row)

    B[i] = slice_dok(B[i], keep_rows, keep_cols)

    if i > 1:
        B[i-1] = slice_dok_columns(B[i-1], keep_rows)

    #B[i] = B[i][:,mask]
    #mask = np.ones(B[i].shape[0], dtype=bool)
    #mask[index_a] = 0
    #if i>1:
    #    B[i-1] = B[i-1][:,mask]
    #B[i] = B[i][mask]

    #####print(B[i].todense())
    return E, B

@profile
def reduce_chain(E,B, maxiter=1e4):
    """
    Parameters:
        E: dict; the collection of all the complexes (which are lists)
        B: dict; the collection of all the boundary matrices (which are sparse matrices)
        i: int; index of the complex to which b belongs
        a,b: lists; the generators that can be reduced
    Returns:
        E: dict; reduced collection of complexes, done in place
        B: dict; reduced collection of boundary matrices, done in place
    """
    iter = 0
    for i in np.sort(list(B.keys()))[::-1]:
        found = True
        
        while found and iter < maxiter:
            # Oss: you waste one step each time you remove all points from the E[i]
            found = False
            for key in B[i].keys():        
                if abs(B[i][key])==1:
                    #print(key) 
                    found = True
                    E, B = pair_reduction(E, B, i, key[0], key[1])
                    break
            iter += 1
        #print(iter, 'stopped at', B[i].todense())
    return E, B

def parallel_ker(B):
    if B.shape[0] == 0:
        sing_vals = np.zeros(B.shape[1])
    elif B.shape[1] == 0:
        sing_vals = np.ones(1)
    else:
        sing_vals = np.zeros(B.shape[1])
        #if B[i].shape[0]>B[i].shape[1]:
        #   sing_vals[i][:min(max_k,B[i].shape[0],B[i].shape[1])] = svds(hstack([B[i].asfptype(), np.zeros([B[i].shape[0],1])]), return_singular_vectors=False, which='SM', k=min(max_k,B[i].shape[0],B[i].shape[1]), solver='propack')
        #else:
        #   sing_vals[i][:min(max_k,B[i].shape[0],B[i].shape[1])] = svds(B[i].asfptype(), return_singular_vectors=False, which='SM', k=min(max_k,B[i].shape[0],B[i].shape[1]), solver='propack')
        #sing_vals[i] = np.zeros(B[i].shape[1]-np.linalg.matrix_rank(B[i].todense()))
        #print(np.linalg.matrix_rank(B[i].todense()))
       #!# removed max_k 
        sing_vals[:min(B.shape[0],B.shape[1])] = np.linalg.svd(B.todense(), compute_uv=False, full_matrices=False)
    return sing_vals

@profile 
def homology_from_reduction(complex, max_consistent=None, maxiter=1e4):
    """
    Given a complex, it returns the number of zero eigenvalues (at most max_k are computed).

    Parameters:
        complex: dict; a dictionary with the list of the n-simplices;
        max_k: int; the maximal number of zeros that will be searched for
        max_consistent: int; it is the number of betti numbers which are expected to be computed correctly. If left as None, it is assumed that there are not k+1-simplices where k is the maximum index of the complex
    Returns:
        betti: list; the first max_consistent betti numbers
    """

    # We build the matrix representation of the border operator for each 
    if max_consistent is None:
        max_consistent = len(complex)-1

    for i in range(min(max_consistent+1, len(complex))):
        if len(complex[i])==0:
            del complex[i]
    
    if max_consistent+1 < len(complex):
        for i in range(max_consistent+1, len(complex)):
            del complex[i]

    B = {i: dok_matrix((len(complex[i-1]), len(complex[i])), dtype=int) for i in range(1, len(complex))}

    for i in range(1,len(complex)): # i=0 leads to a 1-dimensional vector of ones, we will call it when it is needed
        el=0
        for j in complex[i]:
            # we do the decomposition
            boundary = [j[:k]+j[k+1:] for k in range(len(j))]
            signs = [(-1)**k for k in range(len(j))] #not a necessary step
            indices = [complex[i-1].index(k) for k in boundary] # oss: because of the sort in the procedure, we don't need to check for reverse edges
            B[i][indices,el] = signs
            el+=1

    complex, B = reduce_chain(complex, B, maxiter=maxiter)

    # if there are only 1-simplices, it means I eliminated all the other n-simplices, hence I only have separate connected components
    if len(complex)==1:
        return [len(complex[0])]+[0]*(max_consistent-1)

    sing_vals = {}
    
    #res_ker = Parallel(n_jobs=-1)(delayed(parallel_ker)(B[i]) for i in B.keys())
    
    #for i in range(len(res_ker)):
    #    sing_vals[i] = res_ker[i]

    for i in B.keys():
        #print(i, B[i].shape)
        if B[i].shape[0] == 0:
            sing_vals[i] = np.zeros(B[i].shape[1])
        elif B[i].shape[1] == 0:
            sing_vals[i] = np.ones(1)
        else:
            sing_vals[i] = np.zeros(B[i].shape[1])
            #if B[i].shape[0]>B[i].shape[1]:
            #   sing_vals[i][:min(max_k,B[i].shape[0],B[i].shape[1])] = svds(hstack([B[i].asfptype(), np.zeros([B[i].shape[0],1])]), return_singular_vectors=False, which='SM', k=min(max_k,B[i].shape[0],B[i].shape[1]), solver='propack')
            #else:
            #   sing_vals[i][:min(max_k,B[i].shape[0],B[i].shape[1])] = svds(B[i].asfptype(), return_singular_vectors=False, which='SM', k=min(max_k,B[i].shape[0],B[i].shape[1]), solver='propack')
            #sing_vals[i] = np.zeros(B[i].shape[1]-np.linalg.matrix_rank(B[i].todense()))
            #print(np.linalg.matrix_rank(B[i].todense()))
           #!# removed max_k 
            sing_vals[i][:min(B[i].shape[0],B[i].shape[1])] = np.linalg.svd(B[i].todense(), compute_uv=False, full_matrices=False)

    #print('sing_vals',sing_vals)
    #print('c', complex)
    lengths = np.array([len(complex[i]) for i in complex.keys()]+[0]*(max_consistent-len(list(complex.keys()))+1))
    #print(lengths)
    betti = np.zeros(max_consistent, dtype=int)

    # the dimension of the kernel of the maps, the first element is representing d_0, which is not computed
    ker = np.array([len(complex[0])]+[np.count_nonzero(np.isclose(sing_vals[i],np.zeros(len(sing_vals[i])))) for i in sing_vals.keys()]+[0]*(max_consistent-len(sing_vals))) 
    
    #print(ker)
    # betti[0] = dim kernel d[0] - dim image d[1] = dim kernel d[0] - (dim space [1] - dim ker d[1])
    
    betti += ker[:-1]

    betti -= lengths[1:] - ker[1:]
    
    return betti.tolist()

def parallel_digonalization(deltas, i, max_k=10, sparse=False,):
    
    if deltas[i].shape[0] == 1:
        return np.array([deltas[i][0,0]])
    else:
        if sparse:
            return eigsh(deltas[i], k=min(max_k,deltas[i].shape[0]-1), which='SM', return_eigenvectors=False)
        else:
            return  eigh(deltas[i], eigvals_only=True, subset_by_index=[0, min(max_k,deltas[i].shape[0]-1)])

@profile
def homology_from_laplacian(complex, max_k=10, sparse=True):
    """
    Given a complex, it returns the number of zero eigenvalues (at most max_k are computed).

    Parameters:
        complex: dict; a dictionary with the list of the k-simplices for each k in keys (notices that the assumption is that there is no complex of dimension greater than max(k in keys), if this is not true, then the last computed betti number should not be trusted)
        max_k: int; the maximal number of zeros that will be searched for
        sparse: bool; if True, the laplacian matrix will be computed as sparse matrix (this implies that at most min(max_k, n-1) eigenvalues will be computed)
    Returns:
        counts: list; a list where each entrance is the number of 0 eigenvalues of a laplacian matrix
    """

    # We build the matrix representation of the border operator for each 
    max_consistent = len(complex)

    deltas = {}
    for i in range(max_consistent):
        if len(complex[i])>0:
            if sparse:
                deltas[i] = dok_matrix((len(complex[i]), len(complex[i])), dtype=float)
            else:
                deltas[i] = np.zeros((len(complex[i]), len(complex[i])), dtype=float)
        else:
            del complex[i] # if the complex is empty, we don't care about the corresponding laplacian matrix
    #if len(complex[i+1])==0:
    #    del complex[i+1]

    for i in range(1,len(complex)): # i=0 leads to a 1-dimensional vector of ones, we will call it when it is needed
        B = dok_matrix((len(complex[i-1]), len(complex[i])), dtype=int)
        el=0
        for j in complex[i]:
            # we do the decomposition
            boundary = [j[:k]+j[k+1:] for k in range(len(j))]
            signs = [(-1)**k for k in range(len(j))]
            indices = [complex[i-1].index(k) for k in boundary] # oss: because of the sort in the procedure, we don't need to check for reverse edges
            B[indices,el] = signs
            el+=1
        deltas[i-1]+=B@B.T
        deltas[i]+=B.T@B

    eig = {i:[] for i in range(len(deltas)-1)}

    parallel_res = Parallel(n_jobs=-1, prefer='threads')(
        delayed(parallel_digonalization)(deltas, i, max_k, sparse) for i in range(len(deltas))
    )
    for i in range(len(deltas)):
        eig[i] = parallel_res[i]
    #for i in range(len(deltas)):
    #    if deltas[i].shape[0] == 1:
    #        eig[i] = np.array([deltas[i][0,0]])
    #    else:
    #        if sparse:
    #            eig[i] = eigsh(deltas[i], k=min(max_k,deltas[i].shape[0]-1), which='SM', return_eigenvectors=False)
    #        else:
    #            eig[i] = eigh(deltas[i], eigvals_only=True, subset_by_index=[0, min(max_k,deltas[i].shape[0]-1)])
    for i in range(len(deltas), max_consistent):
        eig[i]=eig[i] = np.array([])
    return [np.count_nonzero([np.isclose(eig[i][j],0) for j in range(len(eig[i]))]) for i in range(len(eig))]

@profile
def persistent_homology(points, epsilon_values=None):
    if epsilon_values is None:
        dist_m = distance_matrix(points,points)
        epsilon_values = np.linspace(np.min(dist_m), np.max(dist_m), 10)

    for epsilon in epsilon_values:
        cech = Cech_complex(points, epsilon)
        homology = homology_from_laplacian(cech)

        # We need to keep track of the specific simplices to be able to assign to them a birth and death time
    pass

def radius_selection(points, local=False, ncluster=2):
    """
    Given the set of point we estimate the Bottleneck with en heuristic outlier detection method. The Bottleneck can be used to estimate the condition number.
    
    Parameters:
        points: array-like; an NxD array with position of the datapoints
        local: bool; if True, we estimate for each datapoint a maximal radius, otherwise the estimate is performed with the full histogram
        ncluster: int; the number of clusters to be used in the local estimation, if local is True
    Returns:
        radius: array-like; if local is False it is a float, otherwise an array of floats with the maximum radius at each point
    """

    #### Notice: the agglomerative clustering procedure is very stupid, I am doing it in 1d, in the end I could just consider all the distances up to a value which make the optimal cluster

    distances= distance_matrix(points, points)
    delaunay = Delaunay(points)

    indptr, indices = delaunay.vertex_neighbor_vertices

    distances_hist = np.array([])
    if not local:
        for k in range(len(indptr)-1):
            neigh=indices[indptr[k]:indptr[k+1]]
            distances_hist=np.hstack((distances_hist, distances[k,neigh]))
        kernel = gaussian_kde(distances_hist, bw_method='silverman')
        c=kernel(np.linspace(0,max(distances_hist[1:]),100))
        peaks, _ =find_peaks(c, prominence=0.1)
        if len(peaks) == 0:
            radius = np.mean(distances_hist)
        elif len(peaks) == 1:
            radius = np.linspace(0,max(distances_hist),100)[peaks[0]]
        elif len(peaks) > 1:
            radius = np.linspace(0,max(distances_hist),100)[peaks[1]]/2
        return radius
    else:
        raise NotImplementedError
        #radius = np.zeros(len(points))
        #for k in range(len(indptr)-1):
        #    neigh=indices[indptr[k]:indptr[k+1]]
        #    point_ = distances[k,neigh]
        #    clust = AgglomerativeClustering(n_clusters=ncluster).fit(point_.reshape(-1,1))
        #    l_star = clust.labels_[np.argmin(distances[k,neigh])]
        #    radius[k] = np.max(distances[k,neigh[clust.labels_==l_star]])/2
        #return radius

def bayesian_multinomial_mode(prior, samples, points, size=10000):
    """
    We compute the probability of 'mode' to be the mode of a multinomial distribution. We consider a Dirichlet prior.
    
    Parameters:
        prior: array; an array of the concentration parameters, it must be one item longer than 'points', the last item is the concentration of unseen points
        samples: list; a list containing the samples from the multinomial distribution
        points: list of tuples; a list of the already observed elements
        size: int; number of samples for the Monte Carlo estimate
    Returns:
        probs: array; an array of the probability of each point to be the mode
        var: array; the variance of the single estimates
    """

    assert len(prior) == len(points) + 1
    assert len(prior) >= 2

    if len(prior)==2:
        mode_samples = sum(1 for i in samples if i == points[0])
        rest_samples = len(samples) - mode_samples

        mode_prob = beta.cdf(0.5,prior[0]+mode_samples, prior[1]+rest_samples)

        return [1-mode_prob, mode_prob]

    else:
        posterior = [sum(1 for j in samples if j == points[i]) for i in range(len(points))]
        posterior.append(len(samples) - sum(posterior))
        posterior = np.array(posterior) + prior

        MC_samples = np.random.dirichlet(posterior, size=size)

        maxs = np.argmax(MC_samples, axis=1)

        probs = np.zeros(len(prior))
        var = np.zeros(len(prior))

        counts = np.unique(maxs, return_counts=True)
        probs[counts[0]]+=counts[1]/size

        var = 1/(size-1) * (probs*size*(1-probs)**2+(size-probs*size)*probs**2)

        return probs, var

def reach_estimation(points, NN=10, d=2, n_stochastic=(0,0), method='harmonic_mean', delta=0.1):
    """
    We estimate the reach following Aamari et al. 2019. 
    If we use the method of 'harmonic_mean', we consider the tangent space to be estimated multiple times using PCA on the nearest neighbours of each point. Then an estimator of the distance of the points from the tangent space is built and used to estimate the reach.
    Otherwise we consider for each point n_stochastic estimations of the reach and take some statistic of it. 

    Parameters:
        points: array-like; an NXD array with the dataset
        NN:int; the number of nearest neighbours to be considered to estimate the tangent plane (if n_stochastic[0]>1, then There will be n_stochastic[1] estimations each using int((NN+1)/n_stochastic) or n_stochastic[1] points)
        n_stochastic: tuple (int, int); if greater n_stochastic[0]>1, we estimate the tangent plane n_stochastic[0] times with int((NN+1)/n_stochastic) (or n_stochastic[1] if 'harmonic_mean' is chosen as a method) points
        method: function or string; only considered if n_stochastic[0]>1; if 'harmonic_mean', we estimate the expected value of the reciprocal, otherwise we use the method specified
        delta: float; you should enforce delta sparsity in your dataset; this is the distance thrshold to consider good approximations    
    """
    
    distances = distance_matrix(points, points)**2

    reach=np.inf

    if n_stochastic[0]>1:
        if method == 'harmonic_mean':
            for i in tqdm(range(len(points))):
                Nearest_neigh=np.argsort(distances[i])
                points_ = points-points[i]

                norms = np.zeros((len(points), n_stochastic[0]))
                mask = np.zeros((len(points), n_stochastic[0]))
                mask[i] = np.inf
                for j in range(n_stochastic[0]):
                    random_selection = np.random.choice(Nearest_neigh[:NN+1],n_stochastic[1],replace=False) 
                    
                    X = points_[random_selection]
                    C=1/(n_stochastic[1]-1)*X.T@X
                    _, eigenvectors = np.linalg.eigh(C)
                    projector = eigenvectors[:,:points.shape[1]-d]@eigenvectors[:,:points.shape[1]-d].T
                    points__ = points_@projector # we project directly on the "mixture" of the tangent space 
                    norms[:,j] = 2*np.sum(points__**2, axis=1)**0.5
                    mask[random_selection,j] = np.inf
                norms = (norms+mask)**-1
                
                # We compute the mean of the reciprocals. This estimates the factor (1/2d(x-y,T_xM))
                N_points = np.count_nonzero(norms, axis=1)
                norms = np.sum(norms,axis=1)

                m = np.ones(len(points), dtype=bool)
                m[distances[i]<=delta]=False

                reach=min(reach,min((distances[i,m] * norms[m] / N_points[m])))
        else:
            for i in tqdm(range(len(points))):
                Nearest_neigh=np.argsort(distances[i])
                points_ = points-points[Nearest_neigh[0]]

                random_selection = Nearest_neigh[:NN+1]
                np.random.shuffle(random_selection)

                mask = distances[i]>delta

                reach_=[]
                for j in range(n_stochastic[0]):
                    X = points_[random_selection[j*int((NN+1)/n_stochastic[0]):(j+1)*int((NN+1)/n_stochastic[0])]]
                    C=1/(int(NN+1/n_stochastic[0])-1)*X.T@X
                    _, eigenvectors = np.linalg.eigh(C)
                    projector = eigenvectors[:,:points.shape[1]-d]@eigenvectors[:,:points.shape[1]-d].T
                    points__ = points_@projector # we project directly on the "mixture" of the tangent space 
                    norms = 2*np.sum(points__**2, axis=1)**0.5
                    mask2 = random_selection[j*int((NN+1)/n_stochastic[0]):(j+1)*int((NN+1)/n_stochastic[0])]
                    mask[mask2] = False
                    reach_.append(min((distances[i, mask] / norms[mask])))
                
                reach=min(reach,method(reach_))
                
                
                
    # if n_stochastic[0] is not greater than 1, we estimate the tangent space only once and limit the reach estimation to the points that were not used to estimate the tangent space
    else:
        for i in tqdm(range(len(points))):
            Nearest_neigh=np.argsort(distances[i])
            X = points[Nearest_neigh[:NN+1]]
            points_ = points-X[0]
            X = X-X[0]
            C=1/NN*X.T@X
            _, eigenvectors = np.linalg.eigh(C)
            points_ = points_-(points_@eigenvectors[:,-d:])@eigenvectors[:,-d:].T #we remove the part that is on the tangent space and leave the orthogonal one
            norms = 2*np.sum(points_**2, axis=1)**0.5
            mask = distances[i]>delta
            reach=min(reach,min((distances[i, mask] / norms[mask])))
    return reach

def _make_hessian(par_hessian, n):
    """
    Given a vector of parameters that represent the Hessian of a function, it returns a symmettric matrix.
    The diagonal elements are the last n stored elements.
    """

    H = np.zeros((n,n))

    H[np.triu_indices(n,1)] = par_hessian[:-n]
    H += H.T
    np.fill_diagonal(H, par_hessian[-n:])

    return H

@njit
def make_hessian(par_hessian, n):
    n = int(n)  # ensure integer for Numba
    H = np.zeros((n, n))
    idx = 0

    # Fill upper triangle excluding diagonal
    for i in range(n):
        for j in range(i + 1, n):
            H[i, j] = par_hessian[idx]
            idx += 1

    # Symmetrize by adding transpose
    for i in range(n):
        for j in range(i + 1, n):
            H[j, i] = H[i, j]

    # Fill diagonal
    for i in range(n):
        H[i, i] = par_hessian[-n + i]

    return H

@njit
def der_hessian_indices(points_, n, loss):
    der = np.zeros(n*(n+1)//2)
    index = np.triu_indices(n,1)

    der[:-n]= np.sum(loss[:,np.newaxis]*points_[:,index[0]]*points_[:,index[1]], axis=0)
    der[-n:]= 0.5 * np.sum(loss[:,np.newaxis]*points_**2, axis=0)

    return der

def _least_square_fit(par, points_):
    return 1/len(points_)*np.sum((par[0] + points_@par[1:len(points_[0])+1] + 0.5 * np.einsum('ij,jk,ik->i', points_, make_hessian(par[len(points_[0])+1:], len(points_[0])),  points_ ))**2)

@njit
def quadratic_forms(X, H):
    n = X.shape[0]
    out = np.empty(n)
    for i in range(n):
        x = X[i]
        out[i] = np.dot(x, np.dot(H, x))
    return out

@njit
def least_square_fit(par, points_):
    Hessian_part =  0.5 * quadratic_forms( points_, make_hessian(par[len(points_[0])+1:], len(points_[0])))
    return 1/len(points_)*np.sum((par[0] + points_@par[1:len(points_[0])+1] + Hessian_part)**2)


def _jac_least_square_fit(par, points_):
    loss = (par[0] + points_@par[1:len(points_[0])+1] + 0.5 * np.einsum('ij,jk,ik->i', points_, make_hessian(par[len(points_[0])+1:], len(points_[0])),  points_ ))*2/len(points_)
    der_hessian = der_hessian_indices(points_, len(points_[0]), loss)
    return np.hstack([np.sum(loss),
                      np.sum(loss[:,np.newaxis]*points_, axis=0),
                      der_hessian])

@njit
def jac_least_square_fit(par, points_):
    loss = (par[0] + points_@par[1:len(points_[0])+1] + 0.5 * quadratic_forms(points_, make_hessian(par[len(points_[0])+1:], len(points_[0]))))*2/len(points_)
    der_hessian = der_hessian_indices(points_, len(points_[0]), loss)

    res = np.empty(len(par))
    res[0] = np.sum(loss)
    res[1:len(points_[0])+1] = np.sum(loss[:,np.newaxis]*points_, axis=0)
    res[len(points_[0])+1:] = der_hessian

    return res

@njit
def least_square_fit_jac(par, points_):
    loss = (par[0] + points_@par[1:len(points_[0])+1] + 0.5 * quadratic_forms(points_, make_hessian(par[len(points_[0])+1:], len(points_[0]))))
    
    der_hessian = der_hessian_indices(points_, len(points_[0]), loss*2/len(points_))

    jac = np.empty(len(par))
    jac[0] = np.sum(loss)*2/len(points_)
    jac[1:len(points_[0])+1] = np.sum(loss[:,np.newaxis]*points_, axis=0)*2/len(points_)
    jac[len(points_[0])+1:] = der_hessian

    return 1/len(points_)*np.sum(loss**2), jac

@njit
def _hessian_norm_neg(v, H):
    """
    Given a vector v and a Hessian matrix H, it returns the norm of the Hessian applied to v. We suppose that the gradients are orthonormal.
    
    Parameters:
        v: array-like; a vector of shape (n,)
        H: array-like; an array of symmetric matrix, shape (l,n,n)
    Returns:
        norm: float; the norm of the Hessian applied to v
    """
    norm = 0
    for i in range(len(H)):
        norm += (v.T@H[i]@v)**2
    return -np.sqrt(norm/(v.T@v))

@njit
def hessian_norm_neg(v, H, grads):
    vector = np.zeros(len(H))
    for i in range(len(H)):
        vector[i] = v.T@H[i]@v
    G = grads@grads.T
    G_cross = np.linalg.pinv(G)
    norm = np.sqrt(vector.T@G_cross@vector/(v.T@v))
    return -norm

@njit
def get_local_points(points, distances_row, NN):
    Nearest_neigh = np.argsort(distances_row)
    X = points[Nearest_neigh[:NN+1]]
    X = X - X[0]
    return X

@profile
def max_principal_curvature(points, NN=50, implicit=False, trials=10, d=2, cross_val=None):
    """
    We propose two ways to estimate locally the maximum principal curvature of a point cloud. 
    In one case we approximate locally a chart of the manifold with a Taylor expansion, in the other case we describe the manifold with an implicit representation that is locally approximated with a Taylor expansion.

    Parameters:
        points: array-like; an NxD array with the position of the datapoints
        NN: int; the number of nearest neighbours to be considered to estimate the curvature
        implicit: bool; if True, we estimate the curvature using an implicit representation of the manifold as the locus of zeros of a degree 2 polynomial.
        trials: int or str; the number of initializations used to estimate the minimum of the fit (with SLSQP) or a method for global minimization (for now only 'shgo' is implemented, but it might work only in very small dimensions)
        d:i int; the dimension of the manifold
        cross_val: None or tuple; if None, the standard strategy is applied, if a tuple with two floating numbers, the first is the percentual of points added to NN, the second is the percentual of points used at each iteration. The fitting procedure is performed 'trials' times with a subset of the nearest neighbours points and the best fit is chosen depending on the value of the loss function on the remaining points. Finally, the best fit is used as a starting point for the optimization with the whole set of points.
    """

    distances = distance_matrix(points, points)

    k_max = -np.inf

    if implicit:
        eq_cons = {'type': 'eq',
           'fun' : lambda x: np.sum(x**2)-1,
           'jac' : lambda x: 2*x}
        
        n_constr = len(points[0])-d
        
        """taylor_approximation = f[0] + grad_f@x + 0.5 * x.T@hessian_f@x ->
        par[0] + par[1:len(x)+1]@x+0.5 * x.T @ par[len(x)+1:].reshape(len(x),len(x))@x 
        
        We need par to be minimized and points_ to be a tuple of points
        We know that the hessian is symmetric, and that the function must evaluate 0 at the points
        """

        for i in tqdm(range(len(points))):
            H = np.zeros((n_constr, len(points[0]), len(points[0])))
            grads = np.zeros((n_constr, len(points[0])))
            list_constraints = [eq_cons]
            Nearest_neigh=np.argsort(distances[i])
            if cross_val is not None:
                NN += int(cross_val[0]*NN)

            X = points[Nearest_neigh[:NN+1]]
            X = X-X[0]
            #X = get_local_points(points, distances[i], NN)
            for j in range(n_constr):
                par = np.zeros(len(X[0])*(len(X[0])+1)//2 + len(X[0]) + 1)
                
                if trials == 'shgo':
                    res = shgo(least_square_fit, [(0,1)]*(len(X[0])*(len(X[0])+1)//2 + len(X[0]) + 1), args=(X,), constraints=list_constraints, options={'method': 'SLSQP', 'jac': jac_least_square_fit})
                    par = res.x
                else:
                    if cross_val is None:
                        min_fun = np.inf
                        for k in range(trials):
                            if j==0:
                                par0 = np.random.rand(len(X[0])*(len(X[0])+1)//2 + len(X[0]) + 1)
                            else:
                                par0 = np.random.rand(len(X[0])*(len(X[0])+1)//2 + len(X[0]) + 1) 
                                par0 *= (1-jac_)

                            res = minimize(least_square_fit, par0, args=(X,), method='SLSQP', jac=jac_least_square_fit, constraints=list_constraints)

                            if k == 0 or res.fun < min_fun:
                                min_fun = res.fun
                                par = res.x
                    else:
                        # We use a cross-validation procedure to estimate the best fit
                        min_fun = np.inf
                        for k in range(trials):
                            if j==0:
                                par0 = np.random.rand(len(X[0])*(len(X[0])+1)//2 + len(X[0]) + 1)
                            else:
                                par0 = np.random.rand(len(X[0])*(len(X[0])+1)//2 + len(X[0]) + 1) 
                                par0 *= (1-jac_)

                            # We split the points in two parts, one for the fitting and one for the validation
                            random_selection = np.random.choice(len(X), int(cross_val[1]*len(X)), replace=False)
                            X_fit = X[random_selection]
                            X_val = X[np.setdiff1d(np.arange(len(X)), random_selection)]

                            res = minimize(least_square_fit, par0, args=(X_fit,), method='SLSQP', jac=jac_least_square_fit, constraints=list_constraints)

                            # We compute the loss on the validation set
                            loss_val = least_square_fit(res.x, X_val)

                            if k == 0 or loss_val < min_fun:
                                min_fun = res.fun
                                par = res.x

                        # We use the best fit to estimate the curvature with all the points
                        res = minimize(least_square_fit, par, args=(X,), method='SLSQP', jac=jac_least_square_fit, constraints=list_constraints)
                        par = res.x
                            
                
                # Add one constraint to ensure 'orthogonality' of the implicit functions
                x_rem = np.argmax(np.abs(par)) # remove the biggest element
                jac_ = np.zeros(len(X[0])*(len(X[0])+1)//2 + len(X[0]) + 1)
                jac_[x_rem] = 1
                new_constraint = {'type': 'eq',
                                  'fun': lambda x: x[x_rem],
                                  'jac': lambda x: jac_}
                list_constraints.append(new_constraint)

                H[j] = make_hessian(par[len(X[0])+1:], len(X[0]))
                grads[j] = par[1:len(X[0])+1]
            
            res_2 = shgo(hessian_norm_neg, [(0,1)]*len(X[0]), constraints=eq_cons, args=(H,grads))

            k_max = max(k_max,-res_2.fun)
    else:
        raise NotImplementedError
    
    return k_max

#def _parallel_curvature(points, eq_cons, distances, i=0, NN=50,  trials=10, n_constr=2 ):
#    H = np.zeros((n_constr, len(points[0]), len(points[0])))
#    list_constraints = [eq_cons]
#    Nearest_neigh=np.argsort(distances[i])
#    X = points[Nearest_neigh[:NN+1]]
#    X = X-X[0]
#    for j in range(n_constr):
#        par = np.zeros(len(X[0])*(len(X[0])+1)//2 + len(X[0]) + 1)
#        min_fun = np.inf
#        for k in range(trials):
#            if j==0:
#                par0 = np.random.rand(len(X[0])*(len(X[0])+1)//2 + len(X[0]) + 1)
#            else:
#                par0 += np.random.rand(len(X[0])*(len(X[0])+1)//2 + len(X[0]) + 1) 
#                par0 *= (1-jac_)
#            res = minimize(least_square_fit, par0, args=(X,), method='SLSQP', jac=jac_least_square_fit, constraints=list_constraints)
#            if k == 0 or res.fun < min_fun:
#                min_fun = res.fun
#                par = res.x
#                
#        
#        # Add one constraint to ensure 'orthogonality' of the implicit functions
#        x_rem = np.argmax(np.abs(par)) # remove the biggest element
#        jac_ = np.zeros(len(X[0])*(len(X[0])+1)//2 + len(X[0]) + 1)
#        jac_[x_rem] = 1
#        new_constraint = {'type': 'eq',
#                          'fun': lambda x: x[x_rem],
#                          'jac': lambda x: jac_}
#        list_constraints.append(new_constraint)
#        H[j] = make_hessian(par[len(X[0])+1:], len(X[0]))
#    
#    res_2 = shgo(hessian_norm_neg, [(0.01,1)]*len(X[0]), args=(H,))
#    return -res_2.fun
#
#
#def parallel_max_principal_curvature(points, NN=50, implicit=False, trials=10, d=2):
#    """
#    We parallelize the estimation of the maximum principal curvature.
#    
#    Parameters:
#        points: array-like; an NxD array with the position of the datapoints
#        NN: int; the number of nearest neighbours to be considered to estimate the curvature
#        implicit: bool; if True, we estimate the curvature using an implicit representation of the manifold as the locus of zeros of a degree 2 polynomial.
#        trials: int; the number of initializations used to estimate the minimum of the fit
#        d:i int; the dimension of the manifold
#    """
#    
#    distances = distance_matrix(points, points)
#
#    if implicit:
#        eq_cons = {'type': 'eq',
#           'fun' : lambda x: np.sum(x**2)-1,
#           'jac' : lambda x: 2*x}
#        
#        n_constr = len(points[0])-d
#        
#        """taylor_approximation = f[0] + grad_f@x + 0.5 * x.T@hessian_f@x ->
#        par[0] + par[1:len(x)+1]@x+0.5 * x.T @ par[len(x)+1:].reshape(len(x),len(x))@x 
#        
#        We need par to be minimized and points_ to be a tuple of points
#        We know that the hessian is symmetric, and that the function must evaluate 0 at the points
#        """
#
#        k = Parallel(n_jobs=-1)(
#            delayed(_parallel_curvature)(points, eq_cons, distances, i, NN, trials, n_constr) for i in tqdm(range(len(points)))
#        )
#    else:
#        pass
#
#    return k


### Class for alpha complexes
### Menaging multiple time lags    
### max_principal_curvature from explicit representation, tangent planes from the fitting of the costraints