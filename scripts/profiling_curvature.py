from scipy.integrate import odeint
import numpy as np

def circle(y, t, params=1):
    """Circle ODE system."""
    x, v = y
    dydt = [v, -params*x]
    return dydt

radius = 2.
npoints = 30000
length = 20000
ratio1 = 1.
ratio2 = np.sqrt(2)/2 + np.sqrt(3)/2
tau = 1
max_dim = 5
delta = 5
gen = np.random.Generator(np.random.PCG64(0))
a = gen.normal(size=(2,2))
a /= np.linalg.norm(a, axis=1, keepdims=True)
res=odeint(circle, radius * a[0, :], np.linspace(0, length, npoints), args = (ratio1,))
res2=odeint(circle, radius * a[1,:], np.linspace(0, length, npoints), args = (ratio2,))

observed = np.hstack([res, res2])
observed.shape

from pyclassify.utils import epsilon_sparsification
z = epsilon_sparsification(observed, epsilon=0.5)
z.sum()


from pyclassify.utils import max_principal_curvature_inversion
gen = np.random.default_rng(5)
aamari__, curvature = max_principal_curvature_inversion(points=observed, NN=70, return_all=True, trials=50, cross_val=(0.5,0.9), subset=np.nonzero(z)[0][::10], aamari=True, delta=0.01, low_memory=True, iters_shgo=None, trials_maximization=50, generator=gen)

