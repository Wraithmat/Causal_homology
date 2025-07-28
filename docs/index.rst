.. Topological data analysis for causal discovery documentation master file, created by
   sphinx-quickstart on Mon Jul 28 11:47:20 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.
   
Causal Discovery with Topological Data Analysis
===============================================

We provide the documentation for the functions used to perform **causal discovery**
through **topological data analysis (TDA)**.

The homology groups are computed in several steps:

1. **Build the simplicial complexes** needed for the identification of loops:
   
   a. Using the *smallest ball algorithm* to build the **Čech complex**  
   b. Building the **Vietoris–Rips complex**  
   c. Building the **alpha complex**

2. **Analyse the homology group:**
   
   a. Using the **Laplacian operator**  
   b. Using the **reduction scheme**


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

