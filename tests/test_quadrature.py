from scipy.special import legendre
import torch
import numpy as np
import numpy.testing as npt
import math
from scipy.special import binom

torch.set_default_dtype(torch.float64)
from svise import quadrature


def test_gl_1d():
    def half_circ(x):
        x = x - 1.0
        return torch.sqrt(1 - x.pow(2))

    def npt_quad(n):
        w, x = quadrature.gauss_legendre_vecs(n, a=0.0, b=2.0)
        out = w @ half_circ(x)
        return out.item()

    err_10 = np.abs(math.pi / 2 - npt_quad(10))
    err_20 = np.abs(math.pi / 2 - npt_quad(20))
    err_50 = np.abs(math.pi / 2 - npt_quad(50))
    err_100 = np.abs(math.pi / 2 - npt_quad(100))

    assert err_10 > err_20
    assert err_20 > err_50
    assert err_50 > err_100
    assert err_100 < 1e-6


def test_trapz_1d():
    def half_circ(x):
        x = x - 1.0
        return torch.sqrt(1 - x.pow(2))

    def npt_quad(n):
        w, x = quadrature.trapezoidal_vecs(n, a=0, b=2.0)
        out = w @ half_circ(x)
        return out.item()

    err_10 = np.abs(math.pi / 2 - npt_quad(10))
    err_20 = np.abs(math.pi / 2 - npt_quad(20))
    err_50 = np.abs(math.pi / 2 - npt_quad(50))
    err_100 = np.abs(math.pi / 2 - npt_quad(100))

    assert err_10 > err_20
    assert err_20 > err_50
    assert err_50 > err_100
    assert err_100 < 1e-2


def test_genz_benchmarks():
    from svise.extern import quad_benchmarks

    gbs = quad_benchmarks.genz_benchmarks
    keys = gbs.keys()
    N = int(1e6)
    for k in keys:
        a = gbs[k].lb
        b = gbs[k].ub
        x_mc = torch.rand(N) * (b - a) + a
        I_mc = gbs[k](x_mc).mean() * (b - a)
        err_mgs = "{} benchmark failure".format(k)
        assert np.abs(I_mc.item() - gbs[k].I) < 1e-2, err_mgs
