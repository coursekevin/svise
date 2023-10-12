.. role:: hidden
    :class: hidden-section

:github_url: https://github.com/coursekevin/svise

1D Quadrature rules
=================================
This contains some utilities for defining
1D quadrules used to estimate the integral
over the residual

:hidden:`Gauss legendre quadrature`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: svise.quadrature.gauss_legendre_vecs
    :members:

:hidden:`Trapezoidal quadrature`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: svise.quadrature.trapezoidal_vecs
    :members:

:hidden:`QuadRule1D abstract base class`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: svise.quadrature.QuadRule1D
    :members:

:hidden:`Gauss-Legendre quad rule`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: svise.quadrature.GaussLegendreQuad
    :members:

:hidden:`Unbiased Gauss-Legendre quad rule`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: svise.quadrature.UnbiasedGaussLegendreQuad
    :members:
