import numpy as np
from pyclassify.utils import fast_smallest_ball, Cech_complex, homology_from_laplacian, homology_from_reduction, collapsed_Cech_complex, collapsed_Cech_complex_with_sets

if __name__=='__main__':
    
    np.random.seed(1970)

    n = 100
    points = np.random.uniform(0,1,(n,2))
    C=collapsed_Cech_complex(points, 0.09, 3)
    collapsed_Cech_complex_with_sets(points, 0.09, 3)
    homology_from_laplacian(C, max_k=100, sparse=False)  
    homology_from_reduction(C, max_k=100, max_consistent=3)
