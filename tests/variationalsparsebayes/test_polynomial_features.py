import torch
import numpy.testing as npt
from svise.variationalsparsebayes import *


def test_polynomial_class():
    # test ones edge case
    f = Polynomial([])
    x = torch.randn(20, 2)
    with torch.no_grad():
        npt.assert_array_equal(f(x), torch.ones(20))  # test product
    f = Polynomial([0, 1])
    with torch.no_grad():
        npt.assert_array_equal(f(x), x.prod(-1))
    f = Polynomial([0, 0])
    with torch.no_grad():
        npt.assert_array_equal(f(x), x[..., 0].pow(2))


def test_polynomial_features():
    from sklearn.preprocessing import PolynomialFeatures as PFSklearn

    x = torch.randn(20, 3)
    # compare to sklearn with bias
    f = SparsePolynomialFeatures(3, 5, include_bias=True)
    f_sklearn = PFSklearn(degree=5, include_bias=True)
    with torch.no_grad():
        npt.assert_almost_equal(f(x), f_sklearn.fit_transform(x))
    # compare to sklearn w/o bias
    f = SparsePolynomialFeatures(3, 5, include_bias=False)
    f_sklearn = PFSklearn(degree=5, include_bias=False)
    with torch.no_grad():
        npt.assert_almost_equal(f(x), f_sklearn.fit_transform(x))
    # check that sparsity is being updated correctly
    y_out = f(x)
    f.update_basis(f.sparse_index[::2])
    with torch.no_grad():
        npt.assert_array_equal(y_out[..., ::2], f(x))
    # testing print
    f = SparsePolynomialFeatures(2, 2, include_bias=True)
    assert f.__str__() == "1 + x_0 + x_1 + x_0^2 + x_0 x_1 + x_1^2"
    f = SparsePolynomialFeatures(2, 2, include_bias=False)
    assert f.__str__() == "x_0 + x_1 + x_0^2 + x_0 x_1 + x_1^2"
    f = SparsePolynomialFeatures(2, 2, include_bias=False, input_labels=["x", "y"])
    assert f.__str__() == "x + y + x^2 + x y + y^2"
