from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.functional import softplus
from torch.nn.parameter import Parameter


def inverse_softplus(x: Tensor) -> Tensor:
    return torch.log(torch.exp(x) - 1)


def difference(x1: Tensor, x2: Tensor) -> Tensor:
    r""" Computes the pairwise difference between two pairs of tensors

        :param torch.tensor(n,d) x1: first set of tensors
        :param torch.tensor(m,d) x2: second set of tensors

        :return: squared difference tensor (i,j) corresponds to the  ||x_i - x_j||^2     
        :rtype: torch.tensor(n,m,d)
    """
    if x1.dim() == 1:
        x1 = x1.unsqueeze(-1)
    if x2.dim() == 1:
        x2 = x2.unsqueeze(-1)
    # todo: inplace ops might be faster
    return x1.unsqueeze(-2) - x2.unsqueeze(-3)


class InputWarping(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self, t_in: Tensor, return_grad=False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        pass

    @abstractmethod
    def inverse(self, w: Tensor) -> Tensor:
        pass


class IdentityWarp(InputWarping):
    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, t_in: Tensor, return_grad=False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if return_grad:
            return t_in, torch.ones_like(t_in)
        return t_in

    def inverse(self, w: Tensor) -> Tensor:
        return w


class KumaraswamyWarping(InputWarping):
    _update_buffers: bool = True
    def __init__(
        self,
        t_span: Tuple[Tensor, float],
        eps: float = 1e-6,
        dynamic_update_bounds: bool = True,
    ):
        super(KumaraswamyWarping, self).__init__()
        if not isinstance(t_span[0], Tensor):
            self.register_buffer("t0", torch.tensor(t_span[0]) - eps)
            self.register_buffer("tf", torch.tensor(t_span[1]) + eps)
        else:
            self.register_buffer("t0", (t_span[0] - eps).clone().detach_())
            self.register_buffer("tf", (t_span[1] + eps).clone().detach_())
        self.register_buffer("eps", torch.tensor(eps))
        self.rawalpha = Parameter(torch.Tensor(1))
        self.rawbeta = Parameter(torch.Tensor(1))
        self.dynamic_update_bounds = dynamic_update_bounds
        self.reset_parameters()

    @property
    def dt(self) -> Tensor:
        return self.tf - self.t0

    def update_buffers(self, t_in: Tensor) -> None:
        if self.dynamic_update_bounds:
            if t_in.min() < self.t0:
                self.t0 = (t_in.min() - self.eps).clone().detach_()
            if t_in.max() > self.tf:
                self.tf = (t_in.max() + self.eps).clone().detach_()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.rawalpha, 0.5413)  # -4.0 mean start
        nn.init.constant_(self.rawbeta, 0.5413)

    def get_natural_parameters(self):
        return (softplus(self.rawalpha), softplus(self.rawbeta))

    def forward(
        self, t_in: Tensor, return_grad=False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if self._update_buffers:
            self.update_buffers(t_in)
        alpha, beta = self.get_natural_parameters()
        t = (t_in - self.t0) / self.dt
        diff = 1 - t.pow(alpha)
        w = (1 - diff.pow(beta)).mul(self.dt).add(self.t0)
        if return_grad:
            dw = diff.pow(beta - 1).mul(t.pow(alpha - 1)).mul(beta * alpha)
            return (w, dw)
        else:
            return w

    def inverse(self, w: Tensor) -> Tensor:
        alpha, beta = self.get_natural_parameters()
        w = w.add(-self.t0).div(self.dt)
        t = (1 - (1 - w).pow(1 / beta)).pow(1 / alpha)
        return t.mul(self.dt).add(self.t0)
