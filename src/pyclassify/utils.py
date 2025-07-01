import numpy as np
from scipy.spatial import distance_matrix
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import eigsh, svds
from scipy.linalg import eigh
from line_profiler import profile
from joblib import Parallel, delayed

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
    assert points.shape[0] > 0, "Points should not be empty"
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
        # Check that there exist a point which is within 2-epsilon neighbourhoof of all points in the complex analysed
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
            if links[0] in simplices[0]:
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
            # Check that there exist a point which is within 2-epsilon neighbourhoof of all points in the complex analysed
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
    columns_to_change = B[i][index_a].nonzero()[1]
    rows_to_change = B[i][:,index_b].nonzero()[0]
    
    #print(rows_to_change, columns_to_change)
    #####print('changing', rows_to_change, columns_to_change)
    #####print(B[i][index_a],B[i][index_a].nonzero(), B[i][:,index_b], B[i][:,index_b].nonzero())
    #####print(B[i].todense())
    #####print(B[i-1].todense())

    ##grid = np.meshgrid(rows_to_change, columns_to_change)

    bdba = B[i][index_a,index_b]
    row_ = B[i][index_a, :]
    col_ = B[i][:, index_b]

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

@profile 
def homology_from_reduction(complex, max_k=10, max_consistent=None, maxiter=1e4):
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
def homology(complex):
    pass

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
