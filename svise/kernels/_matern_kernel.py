import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import math
from typing import Tuple, Union
from torch.nn.functional import softplus
from abc import ABC, abstractmethod
from ._utils import (
    inverse_softplus,
    difference,
    InputWarping,
    IdentityWarp,
    KumaraswamyWarping,
)


class Matern52(nn.Module):
    r""" Matern52 kernel extends torch.nn.Module base 
    """

    def __init__(self, input_warping: InputWarping = None, len_init: float = 1.0):
        super(Matern52, self).__init__()
        self.rawlen = Parameter(torch.Tensor(1))
        # having sigf as a parameter helps with training (even though it shouldn't be necessary for a GLM)
        self.rawsigf = Parameter(torch.Tensor(1))
        if input_warping is None:
            self.input_warping = IdentityWarp()
        else:
            self.input_warping = input_warping
        self.reset_parameters(len_init)

    def reset_parameters(self, len_init) -> None:
        r""" Resets the model parameters
        """
        raw_len = inverse_softplus(torch.tensor(len_init))
        nn.init.constant_(self.rawlen, raw_len)
        nn.init.constant_(self.rawsigf, 0.5413)

    def get_natural_parameters(self) -> tuple:
        r""" Returns the parameters in their natural format

            :returns: (exp(loglen), exp(logsigf))
            :type: (torch.tensor, torch.tensor)

        """
        return (softplus(self.rawlen), softplus(self.rawsigf))

    def forward(
        self, x1: Tensor, x2: Tensor, return_grad=False
    ) -> Union[Tensor, Tuple[Tensor]]:
        r""" Computes squared exponential kernel at x1 & x2

            :param torch.tensor(n,d) x1: first set of tensors
            :param torch.tensor(m,d) x2: second set of tensors

            :return: squared exp kernel
            :rtype: torch.tensor(n,m)
        """
        if x1.dim() == 1:
            x1 = x1.unsqueeze(-1)
        if return_grad:
            x1, dx1 = self.input_warping(x1, return_grad=True)
        else:
            x1 = self.input_warping(x1)
        x2 = self.input_warping(x2)
        l, sigf = self.get_natural_parameters()
        mean = x1.mean(0)
        diff = difference(x1 - mean, x2 - mean)
        sqrd = torch.pow(diff, 2).sum(-1)
        # clamp for gradients
        r = sqrd.clamp_min(1e-30).sqrt().mul(math.sqrt(5)).div(l)
        exp_r = torch.exp(-1.0 * r)
        quad = r.pow(2).div(3)
        r = r.add(1)
        sig_sqrd = sigf.pow(2)
        k = exp_r.mul(quad.add(r)).mul(sig_sqrd)
        if return_grad:
            # todo: reuse intermediate quantities better
            dkdt = (
                diff.squeeze(-1)
                .mul(r)
                .mul(exp_r)
                .mul(-5.0 * sig_sqrd)
                .div(3.0 * l.pow(2))
                .mul(dx1)
            )
            return (k, dkdt)
        else:
            return k


class Matern52withGradients(nn.Module):
    r""" Matern52 kernel along with gradient features
    """

    def __init__(self, input_warping: InputWarping = None, len_init: float = 1.0):
        super(Matern52withGradients, self).__init__()
        self.rawlen = Parameter(torch.Tensor(1))
        self.rawsigf = Parameter(torch.Tensor(1))
        if input_warping is None:
            self.input_warping = IdentityWarp()
        else:
            self.input_warping = input_warping
        self.reset_parameters(len_init)

    def reset_parameters(self, len_init) -> None:
        r""" Resets the model parameters
        """
        raw_len = inverse_softplus(torch.tensor(len_init))
        nn.init.constant_(self.rawlen, raw_len)
        nn.init.constant_(self.rawsigf, 0.5413)

    def get_natural_parameters(self) -> tuple:
        r""" Returns the parameters in their natural format

            :returns: (exp(loglen), exp(logsigf))
            :type: (torch.tensor, torch.tensor)

        """
        return (softplus(self.rawlen), softplus(self.rawsigf))

    def forward(
        self, x1: Tensor, x2: Tensor, return_grad=False
    ) -> Union[Tensor, Tuple[Tensor]]:
        r""" Computes matern52 kernel at x1 & x2 with gradient features

            :param torch.tensor(n,d) x1: first set of tensors
            :param torch.tensor(m,d) x2: second set of tensors

            :return: squared exp kernel
            :rtype: torch.tensor(n,m)
        """
        if x1.dim() == 1:
            x1 = x1.unsqueeze(-1)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(-1)
        if return_grad:
            x1, dx1 = self.input_warping(x1, return_grad=True)
        else:
            x1 = self.input_warping(x1)
        x2, dx2 = self.input_warping(x2, return_grad=True)
        l, sigf = self.get_natural_parameters()
        mean = x1.mean(0)
        diff = difference(x1 - mean, x2 - mean)
        sqrd = torch.pow(diff, 2).sum(-1)
        # clamp for gradients
        r = sqrd.clamp_min(1e-30).sqrt().mul(math.sqrt(5)).div(l)
        exp_r = torch.exp(-1.0 * r)
        quad = r.pow(2).div(3)
        r = r.add(1)
        sig_sqrd = sigf.pow(2)
        kttau = exp_r.mul(quad.add(r)).mul(sig_sqrd)
        dkdf = (
            diff.squeeze(-1).mul(r).mul(exp_r).mul(-5.0 * sig_sqrd).div(3.0 * l.pow(2))
        )
        dkdttau = -dkdf.t().mul(dx2).t()
        features = torch.cat([kttau, dkdttau], dim=-1)
        if return_grad:
            dkdttau = dkdf.mul(dx1)
            d2kdtdtau = (
                exp_r.mul(r.add(-3 * quad))
                .mul(5 * sig_sqrd)
                .div(3 * l.pow(2))
                .mul(dx1)
                .mul(dx2.t())
            )
            dfeaturesdt = torch.cat([dkdttau, d2kdtdtau], dim=-1)
            # todo: reuse intermediate quantities better
            return (features, dfeaturesdt)
        else:
            return features


class Matern32(nn.Module):
    r""" Matern32 kernel extends torch.nn.Module base 
    """

    def __init__(self, input_warping: InputWarping = None, len_init: float = 1.0):
        super(Matern32, self).__init__()
        # raise NotImplementedError("Some unit testing is messed up.")
        self.rawlen = Parameter(torch.Tensor(1))
        # having sigf as a parameter helps with training (even though it shouldn't be necessary for a GLM)
        self.rawsigf = Parameter(torch.Tensor(1))
        if input_warping is None:
            self.input_warping = IdentityWarp()
        else:
            self.input_warping = input_warping
        self.reset_parameters(len_init)

    def reset_parameters(self, len_init) -> None:
        r""" Resets the model parameters
        """
        raw_len = inverse_softplus(torch.tensor(len_init))
        nn.init.constant_(self.rawlen, raw_len)
        nn.init.constant_(self.rawsigf, 0.5413)

    def get_natural_parameters(self) -> tuple:
        r""" Returns the parameters in their natural format

            :returns: (exp(loglen), exp(logsigf))
            :type: (torch.tensor, torch.tensor)

        """
        return (softplus(self.rawlen), softplus(self.rawsigf))

    def forward(
        self, x1: Tensor, x2: Tensor, return_grad=False
    ) -> Union[Tensor, Tuple[Tensor]]:
        r""" Computes squared exponential kernel at x1 & x2

            :param torch.tensor(n,d) x1: first set of tensors
            :param torch.tensor(m,d) x2: second set of tensors

            :return: squared exp kernel
            :rtype: torch.tensor(n,m)
        """
        if x1.dim() == 1:
            x1 = x1.unsqueeze(-1)
        if return_grad:
            x1, dx1 = self.input_warping(x1, return_grad=True)
        else:
            x1 = self.input_warping(x1)
        x2 = self.input_warping(x2)
        l, sigf = self.get_natural_parameters()
        mean = x1.mean(0)
        diff = difference(x1 - mean, x2 - mean)
        sqrd = torch.pow(diff, 2).sum(-1)
        # clamp for gradients
        r = sqrd.clamp_min(1e-30).sqrt().mul(math.sqrt(3) / l)
        exp_r = torch.exp(-1.0 * r)
        sig_sqrd = sigf.pow(2)
        k = exp_r.mul(r.add(1)).mul(sig_sqrd)
        if return_grad:
            # todo: reuse intermediate quantities better
            dkdt = diff.squeeze(-1).mul(exp_r).mul(-3.0 * sig_sqrd / l.pow(2)).mul(dx1)
            return (k, dkdt)
        else:
            return k


class Matern12(nn.Module):
    r""" Matern12 kernel extends torch.nn.Module base 
    """

    def __init__(self, input_warping: InputWarping = None, len_init: float = 1.0):
        # raise NotImplementedError("Some unit testing is messed up.")
        super(Matern12, self).__init__()
        self.rawlen = Parameter(torch.Tensor(1))
        # having sigf as a parameter helps with training (even though it shouldn't be necessary for a GLM)
        self.rawsigf = Parameter(torch.Tensor(1))
        if input_warping is None:
            self.input_warping = IdentityWarp()
        else:
            self.input_warping = input_warping
        self.reset_parameters(len_init)

    def reset_parameters(self, len_init) -> None:
        r""" Resets the model parameters
        """
        raw_len = inverse_softplus(torch.tensor(len_init))
        nn.init.constant_(self.rawlen, raw_len)
        nn.init.constant_(self.rawsigf, 0.5413)

    def get_natural_parameters(self) -> tuple:
        r""" Returns the parameters in their natural format

            :returns: (exp(loglen), exp(logsigf))
            :type: (torch.tensor, torch.tensor)

        """
        return (softplus(self.rawlen), softplus(self.rawsigf))

    def forward(
        self, x1: Tensor, x2: Tensor, return_grad=False
    ) -> Union[Tensor, Tuple[Tensor]]:
        r""" Computes squared exponential kernel at x1 & x2

            :param torch.tensor(n,d) x1: first set of tensors
            :param torch.tensor(m,d) x2: second set of tensors

            :return: squared exp kernel
            :rtype: torch.tensor(n,m)
        """
        if x1.dim() == 1:
            x1 = x1.unsqueeze(-1)
        if return_grad:
            x1, dx1 = self.input_warping(x1, return_grad=True)
        else:
            x1 = self.input_warping(x1)
        x2 = self.input_warping(x2)
        l, sigf = self.get_natural_parameters()
        mean = x1.mean(0)
        diff = difference(x1 - mean, x2 - mean)
        sqrd = torch.pow(diff, 2).sum(-1)
        # clamp for gradients
        r = sqrd.clamp_min(1e-30).sqrt().div(l)
        exp_r = torch.exp(-1.0 * r)
        sig_sqrd = sigf.pow(2)
        k = exp_r.mul(sig_sqrd)
        if return_grad:
            dkdt = -diff.squeeze(-1).mul(k).div(r).div(l.pow(2)).mul(dx1)
            return (k, dkdt)

        else:
            return k
