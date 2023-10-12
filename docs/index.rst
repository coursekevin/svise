.. svise documentation master file, created by
   sphinx-quickstart on Fri Nov  4 16:02:40 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/coursekevin/svise

SVISE documentation
================================================
The idea behind this package is that everything
should flow through the :ref:`sde-learner-base-label`
class. After defining an instance of this class, 
you can perform stochastic variational inference 
using any combination of prior and approximate 
posterior. Every :ref:`sde-learner-base-label` 
needs a :ref:`SDE prior`, a :ref:`Diffusion prior`,
an approximate posterior over the state in the 
form of a :ref:`Markov Gaussian process`, a  
:ref:`Likelihood`, and a choice of quad rule 
from the list of :ref:`1D Quadrature rules`.

Some useful extras for solving SDEs defined 
by an :ref:`sde-learner-base-label`, 
solving the Lyapunov equations, and 
computing gradients of a parametrized 
skew-symmetric matrix fast are provided in
the :ref:`Utilities`.

Examples can be found in the `experiments`
directory.




.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. toctree::
   :maxdepth: 2
   :caption: Package reference

   sde_learning
   sde_prior
   diffusion_prior
   markov_gp
   likelihood
   quad 
   utils
   pca




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Research references
===================

* Course, K., Nair, P.B. State estimation of a physical system with unknown governing equations. Nature 622, 261–267 (2023). https://doi.org/10.1038/s41586-023-06574-8 
