import torch
import numpy.testing as npt
from svise.variationalsparsebayes import *


def test_sparse_glm():
    k = SparsePolynomialFeatures(dim=10, degree=2)
    f = SparseGLMGaussianLikelihood(
        d=10, SparseFeatures=k, noise=1e-1, learn_noise=False
    )
    names_f = [n for n in f.parameters()]
    g = SparseGLMGaussianLikelihood(
        d=10, SparseFeatures=k, noise=1e-1, learn_noise=True
    )
    names_g = [n for n in g.parameters()]
    assert len(names_f) + 1 == len(names_g)


if __name__ == "__main__":
    test_sparse_glm()
