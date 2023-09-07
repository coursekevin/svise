from svise.variationalsparsebayes.sparse_glm import SparsePrecomputedFeatures
import torch
import numpy.testing as npt
from svise.variationalsparsebayes import *


def test_precomputed_features():
    d = 20
    f = SparsePrecomputedFeatures(d)
    x = torch.randn(300, d)
    with torch.no_grad():
        npt.assert_array_equal(f(x), x)
    assert f.__str__() == " + ".join(["x_{}".format(j) for j in range(d)])
    # check that sparsity is being updated correctly
    sp_ind = torch.arange(d)[::2]
    f.update_basis(sparse_index=sp_ind)
    with torch.no_grad():
        npt.assert_array_equal(f(x), x[..., sp_ind])
    assert f.__str__() == " + ".join(["x_{}".format(j) for j in sp_ind])

