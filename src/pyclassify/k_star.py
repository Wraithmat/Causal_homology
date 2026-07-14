import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.special import gammaln, binom, logsumexp
from scipy.stats import chi2, kstest
from scipy.stats import binom as binom_dist

alpha=1e-6
maxk = 100

Dthr = chi2.isf(alpha,1)

#@numba.njit(nopython=True, fastmath=True, cache=True)
def compute_kstar(id, N, maxk, distances, dist_indices, alpha=alpha):
    #from dadapy, this implementation has no self neighbourhoods
    kstar = np.empty(N, dtype=int)
    prefactor = np.exp( id / 2.0 * np.log(np.pi) - gammaln((id + 2.0) / 2.0) )
    
    for i in range(N):
        j = 4
        dL = 0.0
        while j < maxk:
            ksel = j - 1 
    
            vvi = prefactor[i] * pow(distances[i, ksel], id[i])
            vvj = prefactor[dist_indices[i, j]] * pow(distances[dist_indices[i, j], ksel], id[dist_indices[i, j]])
            dL = -2.0 * j * ( np.log(vvi) + np.log(vvj) - 2.0 * np.log(vvi + vvj) + np.log(4) ) # ksel -> j; to take into account no self neighbourhoods
            
            if dL > Dthr:
                break
            else:
                j = j + 1
        kstar[i] = j - 1 # fall back to previous iteration where the test had passed

    return kstar

def _compute_binomial_cramerrao(d, k, r, n):
    # from dadapy
    """Calculate the Cramer Rao lower bound for the variance associated with the binomial estimator

    Args:
        d (float): intrinsic dimension
        k (float, np.ndarray(float)): number of neighbours within the external shell
        r (float): ratio among internal and external radii
        n (int): number of points of the dataset

    Returns:
        CramerRao lower bound (float)
    """
    if isinstance(k, np.ndarray):
        k = k.mean()

    return (r ** (-d) - 1) / (k * n * np.log(r) ** 2)

def log_binom_stirling(k, n):
    # from dadapy
    return (
        (k + 0.5) * np.log(k)
        - (n + 0.5) * np.log(n)
        - (k - n + 0.5) * np.log(k - n)
        - 0.5 * np.log(2 * np.pi)
    )


def binomial_loglik(d, k, n, r):
    # from dadapy
    #if isinstance(k, np.ndarray):
    #    pk = np.histogram(k, bins=np.arange(-0.5, k.max() + 1.5))[0]
    #    pk = pk / pk.sum()
    #else:
    #    pk = np.ones(k + 1)
    #    k = [k]
    log_binom = np.log(binom(k, n))
    if np.any(log_binom == np.inf):
        mask = np.where(log_binom == np.inf)[0]
        log_binom[mask] = log_binom_stirling(k[mask], n[mask])
    return -np.sum(
        n * d * np.log(r)
        + (k - n) * np.log(1.0 - r**d)
        + log_binom
        #+ np.log([pk[ki] for ki in k])
    )


def return_ids_kstar_binomial(
            data, maxk=maxk, initial_id=2, Dthr=6.67, r='opt', verbose = True, n_iter = 10, rng = np.random.default_rng(), alpha=1e-6
        ):
            
    # from https://github.com/Fede-stack/Adaptive-nonparametric-dimensionality-reduction/blob/main/src/AdaptiveKLLE.py#L29
    
    """Return the id estimates of the binomial algorithm coupled with the kstar estimation of the scale.

    Args:
        initial_id (float): initial estimate of the id default uses 2NN
        n_iter (int): number of iteration
        Dthr (float): threshold value for the kstar test
        r (float): parameter of binomial estimator, 0 < r < 1
    Returns:
        ids, ids_err, kstars, log_likelihoods
    """
    assert maxk < len(data)

    # start with an initial estimate of the ID
    
    nn = NearestNeighbors(n_neighbors=maxk, n_jobs=-1).fit(data)

    dists, neighs = nn.kneighbors()

    ids = np.zeros((n_iter, len(data)))
    ids_err = np.zeros(n_iter)
    kstars = np.zeros((n_iter, len(data)), dtype=int)
    log_likelihoods = np.zeros(n_iter)
    ks_stats = np.zeros(n_iter)
    p_values = np.zeros(n_iter)


    for i in range(n_iter):
        # compute kstar
        kstar = compute_kstar(
            (np.ones(len(data)) * initial_id if i == 0 else ids[i - 1]),
            len(data),
            maxk,
            dists.astype("float64"),
            neighs.astype("int64"),
            alpha,
        )

        if verbose == True:
            print("iteration ", i)
            print("id ", np.mean(ids[i - 1]) if i > 0 else initial_id)

        # set new ratio
        r_eff = min(0.95,0.2032**(1./(ids[i-1][0] if i > 0 else initial_id))) if r == 'opt' else r
        # compute neighbourhoods shells from k_star
        rk = np.array([dd[kstar[j]] for j, dd in enumerate(dists)])
        rn = rk * r_eff
        n = np.sum([dd < rn[j] for j, dd in enumerate(dists)], axis=1) #numerator for estimate of id
        # compute id
        
        #!!!!! Addition to code
        n = np.maximum(n, 1) # avoid log(0) and log(1)
        
        id = np.log((n.mean() - 1) / (kstar.mean() - 1)) / np.log(r_eff)
        # compute id error
        id_err = _compute_binomial_cramerrao(id, kstar-1, r_eff, len(data))
        # compute likelihood
        #return(kstar, n, r_eff)
        log_lik = binomial_loglik(id, kstar - 1, n - 1, r_eff)
        # model validation through KS test
        n_model = binom_dist(kstar-1, r_eff**id)
        ks, pv = kstest(n-1, n_model.cdf)

        ids[i] = id*np.ones(len(data))
        ids_err[i] = id_err
        kstars[i] = kstar
        log_likelihoods[i] = log_lik
        ks_stats[i] = ks
        p_values[i] = pv

    
    intrinsic_dim_scale = 0.5 * (rn.mean() + rk.mean())

    return ids, kstars[(n_iter - 1), :], ids_err, log_likelihoods, ks_stats, p_values, intrinsic_dim_scale

def return_local_ids_kstar(
            data, maxk=maxk, initial_id=2, Dthr=6.67, r='opt', verbose = True, n_iter = 10, rng = np.random.default_rng(), alpha=1e-6
        ):
                
    """Return the id estimates of the binomial algorithm coupled with the kstar estimation of the scale.

    Args:
        initial_id (float): initial estimate of the id default uses 2NN
        n_iter (int): number of iteration
        Dthr (float): threshold value for the kstar test
        r (float): parameter of binomial estimator, 0 < r < 1
    Returns:
        ids, ids_err, kstars, log_likelihoods
    """
    assert maxk < len(data)

    # start with an initial estimate of the ID
    
    nn = NearestNeighbors(n_neighbors=maxk, n_jobs=-1).fit(data)

    dists, neighs = nn.kneighbors()

    ids = np.zeros((n_iter, len(data)))
    kstars = np.zeros((n_iter, len(data)), dtype=int)

    for i in range(n_iter):
        # compute kstar
        kstar = compute_kstar(
            (np.ones(len(data)) * initial_id if i == 0 else ids[i - 1]),
            len(data),
            maxk,
            dists.astype("float64"),
            neighs.astype("int64"),
            alpha,
        )

        if verbose == True:
            print("iteration ", i)
            print("id ", (np.mean(ids[i - 1]), np.min(ids[i - 1]), np.max(ids[i - 1])) if i > 0 else initial_id)

        # compute neighbourhoods shells from k_star
        ids[i] = np.array([1/(kstar[j]-2)*np.sum(np.log(dists[j,kstar[j]])-np.log(dists[j,:kstar[j]-1])) for j in range(len(kstar))])**-1
        kstars[i] = kstar
        

    return ids, kstars[(n_iter - 1), :]



def find_Kstar_neighs(self, kstars):
    # from https://github.com/Fede-stack/Adaptive-nonparametric-dimensionality-reduction/blob/main/src/AdaptiveKLLE.py#L29

    """
    Finds the k* nearest neighbors for each point in the dataset, where k* varies per point.
    
    Args:
        kstars (numpy.ndarray): Array containing the number of neighbors (k*) to find for each point.
                               Length should match the number of points in self.X.
    
    Returns:
        list: List of lists where each inner list contains the indices of the k* nearest 
              neighbors for the corresponding point, excluding the point itself.
    """
    nn = NearestNeighbors(n_jobs=-1)
    nn.fit(self.X)

    neighs_ind = []
    for i, obs in enumerate(self.X):
        distance, ind = nn.kneighbors([obs], n_neighbors=kstars[i] + 1)

        k_neighs = ind[0][1:]
        neighs_ind.append(k_neighs.tolist())
    return neighs_ind
