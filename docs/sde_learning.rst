.. role:: hidden
    :class: hidden-section

:github_url: https://github.com/coursekevin/svise

High-level interfaces
=================================
This page contains a reference of some convenience 
classes for performing state estimation with 
stochastic variational inference (SVISE). Inheriting
from the SDELearner class will handle most things 
common to all potential parametrizations of the 
prior and approximate posterior such as computing 
the loss, resampling, etc.

.. _sde-learner-base-label:

:hidden:`SDELearner`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: svise.sde_learning.SDELearner
    :members:

:hidden:`Sparse polynomial SDEs`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: svise.sde_learning.SparsePolynomialSDE
    :members:

:hidden:`Neural SDE`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: svise.sde_learning.NeuralSDE
    :members:

:hidden:`Sparse polynomial for SDEs with Newtonian drift`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: svise.sde_learning.SparsePolynomialIntegratorSDE
    :members:

:hidden:`State estimation`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: svise.sde_learning.StateEstimator
    :members:

