from typing import Callable, List, Union, Tuple
import torch
import numpy as np
from torch import Tensor
from torch.fft import fft
import math
from ..extern import fastgl


def gauss_legendre_vecs(
    N: int,
    a: float = -1.0,
    b: float = 1.0,
    dtype: torch.dtype = None,
) -> Tuple[Tensor, Tensor]:
    """
    Returns the gauss-legendre quadrature weights and nodes provided
    by fastgl (see svise/extern for LICENSE and details)

    Args:
        N (int): Number of integration nodes
        dtype (torch.dtype, optional): Return data type. Defaults to torch.float32.

    Returns:
        (dtype, dtype): integration weights, integration nodes
    """
    if dtype == None:
        dtype = torch.get_default_dtype()
    x = []
    w = []
    for j in range(1, N + 1):
        _, wt, xt = fastgl.glpair(N, j)
        w.append(wt)
        x.append(xt)
    wq = torch.tensor(w, dtype=dtype) * (b - a) / 2
    xq = torch.tensor(x, dtype=dtype) * (b - a) / 2 + (a + b) / 2
    return (wq, xq)


def trapezoidal_vecs(
    N: int,
    a: float = -1.0,
    b: float = 1.0,
    dtype: torch.dtype = None,
) -> Tuple[Tensor, Tensor]:
    """
    Returns the trapezoidal qudrature weights and nodes

    Args:
        N (int): Number of integration nodes
        a (float, optional): Start of interval. Defaults to -1.0.
        b (float, optional): End of interval. Defaults to 1.0.
        dtype (torch.dtype, optional): Return data type. Defaults to torch.float32.

    Returns:
        (dtype, dtype): integration weights, integration nodes
    """
    if dtype == None:
        dtype = torch.get_default_dtype()
    x = torch.linspace(a, b, N, dtype=dtype)
    w = torch.ones(N, dtype=dtype)
    w[0] = w[0] * 0.5
    w[-1] = w[-1] * 0.5
    w = (b - a) / (N - 1) * w
    return (w, x)
