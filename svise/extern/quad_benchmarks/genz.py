import torch
from torch import Tensor
import math
from abc import ABC, abstractmethod, ABCMeta


def _compute_coefficients(d: int) -> Tensor:
    i = torch.arange(d) + 1
    ci = 10 ** (-15 * (i / d).pow(2))
    ci = ci * (ci.sum()).pow(-1)
    return ci


class QuadBenchmark1D(ABC):
    def __init__(self) -> None:
        pass

    @property
    @abstractmethod
    def lb(self):
        pass

    @property
    @abstractmethod
    def ub(self):
        pass

    @property
    @abstractmethod
    def I(self):
        pass

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        pass


C_VAL = 5.0


class CornerPeak(QuadBenchmark1D):
    lb = 0.0
    ub = 1.0
    c = C_VAL
    I = 1 / (c + 1)

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: Tensor) -> Tensor:
        return (1 + (self.c * x)).pow(-1 * (1 + 1))


class Oscillatory(QuadBenchmark1D):
    lb = -1.0
    ub = 1.0
    c = C_VAL
    I = 2 * math.sin(c) / c

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: Tensor) -> Tensor:
        return torch.cos((self.c * x))


class GaussianPeak(QuadBenchmark1D):
    lb = -1.0
    ub = 1.0
    c = C_VAL
    I = math.sqrt(math.pi) * math.erf(c) / c

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: Tensor) -> Tensor:
        return torch.exp(-(self.c ** 2) * x.pow(2))


class Continuous(QuadBenchmark1D):
    lb = -1.0
    ub = 1.0
    c = C_VAL
    I = (2 - 2 * math.exp(-c)) / c

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: Tensor) -> Tensor:
        return torch.exp(-self.c * torch.abs(x))


class ProductPeak(QuadBenchmark1D):
    lb = -1.0
    ub = 1.0
    c = C_VAL
    I = 2 * math.atan(1 / c) / c

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: Tensor) -> Tensor:
        return (self.c ** 2 + x.pow(2)).pow(-1)


class Discontinuous(QuadBenchmark1D):
    lb = -1.0
    ub = 1.0
    c = C_VAL
    I = (1 - math.exp(-c)) / c

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, x: Tensor) -> Tensor:
        f = torch.exp(self.c * x)
        f[(x > 0) > 0] = 0.0
        return f


def corner_peak(x: Tensor) -> Tensor:
    if len(x.shape) == 1:
        d = 1
        x = x.unsqueeze(1)
    else:
        d = x.shape[1]
    ci = _compute_coefficients(d)
    return (1 + (ci * x).sum(-1)).pow(-1 * (d + 1))


def oscillatory(x: Tensor) -> Tensor:
    if len(x.shape) == 1:
        d = 1
        x = x.unsqueeze(1)
    else:
        d = x.shape[1]
    omega = 0.0
    ci = _compute_coefficients(d)
    return torch.cos(2 * math.pi * omega + (ci * x).sum(-1))


def gaussian_peak(x: Tensor) -> Tensor:
    if len(x.shape) == 1:
        d = 1
        x = x.unsqueeze(1)
    else:
        d = x.shape[1]
    ci = _compute_coefficients(d)
    wi = torch.zeros(d)
    return torch.exp(-(ci.pow(2) * (x - wi).pow(2)).sum(-1))


def continuous(x: Tensor) -> Tensor:
    if len(x.shape) == 1:
        d = 1
        x = x.unsqueeze(1)
    else:
        d = x.shape[1]
    ci = _compute_coefficients(d)
    wi = torch.zeros(d)
    return torch.exp(-ci * torch.abs(x - wi).sum(-1))


def product_peak(x: Tensor) -> Tensor:
    if len(x.shape) == 1:
        d = 1
        x = x.unsqueeze(1)
    else:
        d = x.shape[1]
    ci = _compute_coefficients(d)
    wi = torch.zeros(d)
    return (ci.pow(-2) + (x - wi).pow(2)).sum(-1).pow(-1)


def discontinuous(x: Tensor) -> Tensor:
    if len(x.shape) == 1:
        d = 1
        x = x.unsqueeze(1)
    else:
        d = x.shape[1]
    ci = _compute_coefficients(d)
    wi = torch.zeros(d)
    f = torch.exp((ci * x).sum(-1))
    f[(x > 0).sum(-1) > 0] = 0.0
    return f


genz_benchmarks = {
    "corner_peak": CornerPeak(),
    "oscillatory": Oscillatory(),
    "gaussian_peak": GaussianPeak(),
    "continuous": Continuous(),
    "product_peak": ProductPeak(),
    "discontinuous": Discontinuous(),
}
