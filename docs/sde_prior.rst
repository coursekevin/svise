.. role:: hidden
    :class: hidden-section

:github_url: https://github.com/coursekevin/svise

SDE prior 
=================================
This section contains some utilties for 
defining priors over the drift function 
in the prior over the state. All utilties
defined after the SDEPrior base class inherit
from this class.

:hidden:`SDEPrior base class`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: svise.sde_learning.SDEPrior
    :members:

:hidden:`Exact motion model`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: svise.sde_learning.ExactMotionModel
    :members:

:hidden:`Sparse linear model`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: svise.sde_learning.SparseMultioutputGLM
    :members:

:hidden:`Second order sparse linear model`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: svise.sde_learning.SparseIntegratorGLM
    :members:

:hidden:`Sparse linear model of neighbours`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: svise.sde_learning.SparseNeighbourGLM
    :members:
