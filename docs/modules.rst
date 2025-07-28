Betti Number Estimation from Point Clouds
=================================================

Homology computations
----------------------
We provide functions to compute the homology of simplicial complexes using the Laplacian or reduction methods.

.. autofunction:: pyclassify.utils.homology_from_laplacian
.. autofunction:: pyclassify.utils.homology_from_reduction

Reach estimations
------------------
We provide functions to estimate the reach of a point cloud.

.. autofunction:: pyclassify.utils.reach_estimation
.. autofunction:: pyclassify.utils.radius_selection
.. autofunction:: pyclassify.utils.max_principal_curvature

Complexes construction
-------------------------------
We support the construction of simplicial complexes from point clouds, including Vietoris-Rips, Cech complexes and alpha complexes.

.. autofunction:: pyclassify.utils.Vietoris_Rips_complex
.. autofunction:: pyclassify.utils.Cech_complex
.. autofunction:: pyclassify.utils.collapsed_Cech_complex
.. autofunction:: pyclassify.utils.collapsed_Cech_complex_with_sets
.. autofunction:: pyclassify.utils.alpha_complex

Others
---------
.. autofunction:: pyclassify.utils.bayesian_multinomial_mode