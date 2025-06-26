import numpy as np
from pyclassify.utils import fast_smallest_ball, Cech_complex, homology_from_laplacian

if __name__=='__main__':
    
    np.random.seed(1970)

    n = 1000
    points = np.random.uniform(0,1,(n,2))
    C=Cech_complex(points, 0.01, 3)
    homology_from_laplacian(C, sparse=False)
