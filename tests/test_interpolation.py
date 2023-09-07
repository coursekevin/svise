import matplotlib.pyplot as plt
import numpy.testing as npt
from scipy.interpolate import BarycentricInterpolator
import torch
torch.set_default_dtype(torch.float64)

from svise import quadrature



def test_barycentric_interpolate():
    _, xi = quadrature.gauss_legendre_vecs(10)
    yi = torch.sin(xi)
    scipy_interp = BarycentricInterpolator(xi, yi)
    torch_interp = quadrature.BarycentricInterpolate(xi, yi)
    x_eval = torch.linspace(-1, 1, 300)
    with torch.no_grad():
        npt.assert_array_almost_equal(torch_interp(x_eval), scipy_interp(x_eval))


if __name__ == "__main__":
    test_barycentric_interpolate()

