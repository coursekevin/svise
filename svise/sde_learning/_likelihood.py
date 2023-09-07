import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor
import math
from typing import Union, Callable, List
from abc import ABC, abstractmethod
from ..utils import *
from ._marginal_sde import *

__all__ = [
    "Likelihood",
    "IndepGaussLikelihood",
    "IndepGaussLikelihoodList",
]


class Likelihood(ABC, nn.Module):
    """Abstract base class defining the likelihood of
    the observations given specifc state.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def mean_log_likelihood(
        self,
        t: Tensor,
        x: Tensor,
        marginal_sde: MarginalSDE,
        n_reparam_samples: int,
        **kwargs,
    ) -> Tensor:
        """
        Computes the mean log-likelihood for a batch of data. When closed form is not available,
        this is approximated by sampling from the marginal SDE.

        Args:
            t (Tensor): (bs, ) time stamps of measurements
            x (Tensor): (bs, D) measurements
            marginal_sde (MarginalSDE): marginal stochatic differential equationdefining the approximate posterior
            n_reparam_samples (int): number of reparam samples to use when closed form is not available (ignored otherwise)

        Returns:
            Tensor: mean log-likelihood for batch of data
        """
        pass

    @abstractmethod
    def pretrain_log_like(
        self, t: Tensor, x: Tensor, marginal_sde: MarginalSDE
    ) -> Tensor:
        """Loss used to pretrain marginal SDE
            t (Tensor): time stamps
            x (Tensor): observations
            marginal_sde (MarginalSDE): Markov GP definition

        Returns:
            Tensor: log-likelihood of observations
        """
        pass


class IndepGaussLikelihood(Likelihood):
    """
    Likelihood for the case that the observations are drawn from a diagonal Gaussian.

    Args:
        g (Union[Callable, Tensor]): observation function / matrix
        d (int): number of states
        measurement_noise (Tensor): observation variance

    """

    def __init__(
        self,
        g: Union[Callable, Tensor],
        d: int,
        measurement_noise: Tensor,
    ) -> None:
        super().__init__()
        self.register_buffer("log2pi", torch.log(torch.tensor(2 * math.pi)))
        if isinstance(g, Callable):
            self.measurement_type = "nonlinear"
            self.g = g
        elif isinstance(g, Tensor):
            self.measurement_type = "linear"
            assert g.shape[0] == d
            self.register_buffer("g", g)
        else:
            raise ValueError("g must be either a Callable or a Tensor")
        self.d = d
        self.register_buffer("measurement_noise", measurement_noise)
        assert len(measurement_noise) == d

    def mean_log_likelihood(
        self,
        t: Tensor,
        x: Tensor,
        marginal_sde: MarginalSDE,
        n_reparam_samples: int,
    ) -> Tensor:
        """
        Computes the mean log-likelihood for a batch of data. When g is nonlinear,
        this is approximated by sampling from the marginal SDE.

        Args:
            t (Tensor): (bs, ) time stamps of measurements
            x (Tensor): (bs, D) measurements
            marginal_sde (MarginalSDE): marginal stochatic differential equation defining the approximate posterior
            n_reparam_samples (int): number of reparam samples to use when closed form is not available (ignored otherwise)

        Returns:
            Tensor: mean log-likelihood for batch of data
        """
        var = self.measurement_noise
        if self.measurement_type == "linear":
            m = marginal_sde.mean(t)
            # compute updated mean and covariance correction term
            if marginal_sde.low_rank_cov:
                R1, D1 = marginal_sde.K_decomp(t)
                gr1 = self.g @ R1
                cov_corr = (
                    (
                        gr1.transpose(-2, -1).div(var)
                        * gr1.mul(D1.unsqueeze(-2)).transpose(-2, -1)
                    )
                    .sum((-1, -2))
                    .mean(0)
                )
            else:
                K = marginal_sde.K(t)
                cov_corr = (self.g.t().div(var) @ self.g).mul(K).sum((-1, -2)).mean(0)

            sqrd_diff = ((x - m @ self.g.t()).pow(2) / var).sum(-1).mean(0)
            log_like = (
                -0.5 * sqrd_diff
                - 0.5 * cov_corr
                - self.d / 2 * self.log2pi
                - 0.5 * torch.log(var).sum()
            )
        elif self.measurement_type == "nonlinear":
            zs = marginal_sde.generate_samples(t, n_reparam_samples)
            mu = self.g(t, zs)
            log_like = mean_diagonal_gaussian_loglikelihood(
                x, mu, var, log2pi=self.log2pi
            )
        else:
            raise ValueError("measurement_type must be either linear or nonlinear")
        return log_like

    def pretrain_log_like(
        self, t: Tensor, x: Tensor, marginal_sde: MarginalSDE
    ) -> Tensor:
        var = self.measurement_noise
        m = marginal_sde.mean(t)
        if self.measurement_type == "linear":
            # compute updated mean and covariance correction term
            sqrd_diff = ((x - m @ self.g.t()).pow(2) / var).sum(-1)
            log_like = mean_diagonal_gaussian_loglikelihood(
                x, m @ self.g.t(), var, log2pi=self.log2pi
            )
        elif self.measurement_type == "nonlinear":
            raise NotImplementedError("pretraining not implemented for nonlinear g")
            log_like = mean_diagonal_gaussian_loglikelihood(
                x, self.g(t, m), var, log2pi=self.log2pi
            )
        else:
            raise ValueError("measurement_type must be either linear or nonlinear")
        return log_like


class IndepGaussLikelihoodList(Likelihood):
    """DEPRECATED"""

    def __init__(self, g: List, d: List, measurement_noise: List) -> None:
        super().__init__()
        likelihood_list = []
        for i, (g_i, d_i, m_i) in enumerate(zip(g, d, measurement_noise)):
            assert g_i.shape[0] == d_i, "g and d must have same shape"
            likelihood = IndepGaussLikelihood(g_i, d_i, m_i)
            likelihood_list.append(likelihood)
        self.likelihood_list = nn.ModuleList(likelihood_list)

    def mean_log_likelihood(
        self, t: List, x: List, marginal_sde, n_reparam_samples
    ) -> Tensor:
        num_lls = len(self.likelihood_list)
        assert num_lls == len(x), "x must have same length as g"
        assert num_lls == len(t), "t must have same length as g"
        log_like = 0.0
        for likelihood, t_i, x_i in zip(self.likelihood_list, t, x):
            log_like += likelihood.mean_log_likelihood(
                t_i, x_i, marginal_sde, n_reparam_samples
            )
        return log_like

    def pretrain_log_like(self, t: List, x: List, marginal_sde) -> Tensor:
        num_lls = len(self.likelihood_list)
        assert num_lls == len(x), "x must have same length as g"
        assert num_lls == len(t), "t must have same length as g"
        log_like = 0.0
        for likelihood, t_i, x_i in zip(self.likelihood_list, t, x):
            log_like += likelihood.pretrain_log_like(t_i, x_i, marginal_sde)
        return log_like
