from .svi_half_cauchy import SVIHalfCauchyPrior
from .sparse_glm import softplus, inverse_softplus
import torch
from torch.nn import ReLU
import torch.nn as nn
from torch.nn import init
from torch import Tensor
from torch.nn.parameter import Parameter
import math
from typing import Callable, Union, Tuple


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class BayesianLinear(nn.Module):
    """
    Linear layer for sparse Bayesian neural networks with a half-cauchy prior.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tau: Union[Tensor, float] = 1e-5,
        n_reparam: int = 20,
        nonlinearity: Callable[[Tensor], Tensor] = ReLU(),
    ):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_reparam = n_reparam
        w, b = self.init_parameters()
        weights_init = torch.cat([w.flatten(), b])
        self.prior = SVIHalfCauchyPrior(
            d=(in_features * out_features + out_features), tau=tau, w_init=weights_init
        )
        self._wtilde = self.prior.get_reparam_weights(n_reparam)
        self.nonlinearity = nonlinearity

    def forward(self, x: Tensor) -> Tensor:
        return self.nonlinearity(x @ self.w.transpose(-2, -1) + self.b.unsqueeze(1))

    def reparam_sample(self, n: int = None) -> None:
        if n is None:
            n = self.n_reparam
        self._wtilde = self.prior.get_reparam_weights(n)

    def init_parameters(self) -> Tuple[Tensor, Tensor]:
        w = init.kaiming_uniform_(
            torch.empty(self.out_features, self.in_features), a=math.sqrt(5)
        )
        fan_in, _ = init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        b = init.uniform_(torch.empty(self.out_features), -bound, bound)
        return (w, b)

    @property
    def w(self) -> Tensor:
        return self._wtilde[:, : self.in_features * self.out_features].reshape(
            -1, self.out_features, self.in_features
        )

    @property
    def b(self) -> Tensor:
        return self._wtilde[:, self.in_features * self.out_features :]

    @property
    def kl_divergence(self) -> Tensor:
        return self.prior.kl_divergence()


class BayesianResidual(BayesianLinear):
    """
    Residual layer for sparse Bayesian neural networks with a half-cauchy prior.
    """

    def __init__(
        self,
        in_features: int,
        tau: Union[Tensor, float] = 1e-5,
        n_reparam: int = 20,
        nonlinearity: Callable[[Tensor], Tensor] = ReLU(),
    ):
        super().__init__(
            in_features,
            in_features,
            tau=tau,
            n_reparam=n_reparam,
            nonlinearity=nonlinearity,
        )

    def forward(self, x: Tensor) -> Tensor:
        return super().forward(x) + x


class SparseBNN(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_hidden: int = 200,
        n_layers: int = 5,
        tau: Union[Tensor, float] = 1e-5,
        n_reparam: int = 20,
        nonlinearity: Callable[[Tensor], Tensor] = ReLU(),
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_reparam = n_reparam
        layers = [
            BayesianLinear(
                in_features,
                n_hidden,
                tau=tau,
                n_reparam=n_reparam,
                nonlinearity=nonlinearity,
            ),
        ]
        layers += (n_layers - 1) * [
            BayesianResidual(
                n_hidden, tau=tau, n_reparam=n_reparam, nonlinearity=nonlinearity
            )
        ]
        layers += [
            BayesianLinear(
                n_hidden,
                out_features,
                tau=tau,
                n_reparam=n_reparam,
                nonlinearity=Identity(),
            ),
        ]
        self.layers = nn.Sequential(*layers)
        self.sigma = torch.ones(out_features)

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)

    def log_likelihood(self, x: Tensor, y: Tensor) -> Tensor:
        sigma = self.sigma
        y_pred = self.forward(x)
        return (
            -0.5 * ((y - y_pred).pow(2) / self.sigma.pow(2)).sum(-1).mean()
            - self.out_features / 2 * math.log(2 * math.pi)
            - 0.5 * torch.log(sigma.pow(2).prod())
        )

    def elbo(self, x: Tensor, y: Tensor, n_data: int, beta: float = 1.0) -> Tensor:
        return self.log_likelihood(x, y) - (beta / n_data) * self.kl_divergence

    def reparam_sample(self, n: int = None) -> None:
        if n is None:
            n = self.n_reparam
        for layer in self.layers:
            layer.reparam_sample(n)

    @property
    def kl_divergence(self) -> Tensor:
        kl_divergence = 0.0
        for layer in self.layers:
            kl_divergence += layer.kl_divergence
        return kl_divergence

    @property
    def sigma(self) -> Tensor:
        return softplus(self.__raw_sigma)

    @sigma.setter
    def sigma(self, sigma: Tensor) -> None:
        self.__raw_sigma = nn.Parameter(inverse_softplus(sigma))
        # self.__raw_sigma = inverse_softplus(sigma)

