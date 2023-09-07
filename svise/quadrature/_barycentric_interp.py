import torch
import torch.nn as nn
from torch import Tensor
import math


class BarycentricInterpolate(nn.Module):
    """
    Polynomial interpolant based on scipy implementation

    """

    def __init__(self, x: Tensor, f: Tensor) -> None:
        super().__init__()
        self.n = len(x)
        self.dtype = x.dtype
        assert self.n == len(f), "Interpolation requires x and f to be the same length."
        self.register_buffer("xi", x)
        self.register_buffer("fi", f)
        self.register_buffer(
            "wi", torch.zeros(self.n, dtype=self.dtype, device=x.device)
        )
        # scipy implementation for computing weights
        self.wi[0] = 1
        for j in range(1, self.n):
            self.wi[:j] *= self.xi[j] - self.xi[:j]
            self.wi[j] = (self.xi[:j] - self.xi[j]).prod()
        self.wi = self.wi.pow(-1)

    def forward(self, x: Tensor) -> Tensor:
        c = x.unsqueeze(1) - self.xi.unsqueeze(0)
        z = c == 0
        c[z] = 1.0
        c = c.pow(-1)
        r = torch.nonzero(z, as_tuple=True)
        numer = c @ (self.fi * self.wi)
        denom = c @ self.wi
        out = numer / denom
        out[r[0]] = self.fi[r[1]]
        return out

