import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor
from typing import List, Union, Tuple, Callable, Optional
from abc import ABC, abstractmethod
from ..utils import *
from ..quadrature._unbiased_quadrature import QuadRule1D
from ..quadrature import UnbiasedGaussLegendreQuad
from ._marginal_sde import *
from ._diffusion_prior import *
from ._likelihood import *
from ._sde_prior import *
from ..variationalsparsebayes.sparse_glm import SparsePolynomialFeatures

__all__ = [
    "SDELearner",
    "SparsePolynomialSDE",
    "SparsePolynomialIntegratorSDE",
    "StateEstimator",
    "NeuralSDE",
]


class SDELearner(nn.Module):
    """Base class for combining different priors and likelihoods for performing
    simultaneous state and governing equation discovery.

    Args:
        marginal_sde (MarginalSDE): Parameterization for the approximate Markov GP
        likelihood (Likelihood): Likelihood for the observations
        quad_scheme (QuadRule1D): 1D quadrature rule used to approximate the residual
        sde_prior (SDEPrior): The prior over the drift function
        diffusion_prior (DiffusionPrior): The prior over the diffusion matrix
        n_reparam_samples (int): The number of samples to use when doing the reparametrization trick
    """

    def __init__(
        self,
        marginal_sde: MarginalSDE,
        likelihood: Likelihood,
        quad_scheme: QuadRule1D,
        sde_prior: SDEPrior,
        diffusion_prior: DiffusionPrior,
        n_reparam_samples: int,
    ) -> None:
        super().__init__()
        self.marginal_sde = marginal_sde
        self.likelihood = likelihood
        self.quad_scheme = quad_scheme
        self.sde_prior = sde_prior
        self.diffusion_prior = diffusion_prior
        self.n_reparam_samples = n_reparam_samples
        assert len(self.state_params()) + len(self.sde_params()) == len(
            list(self.parameters())
        ), "State and SDE parameters not added correctly."

    def state_params(self) -> List[Parameter]:
        """Returns parameters related to only the state estimator

        Returns:
            List[Parameter]: parameters related to the state estimator
        """
        return list(
            set(self.marginal_sde.parameters()) - set(self.diffusion_prior.parameters())
        )

    def sde_params(self) -> List[Parameter]:
        """Returns all parameters not related to the state estimate

        Returns:
            List[Parameter]: parameters not related to the state estimate
        """
        return list(set(self.parameters()) - set(self.state_params()))

    def residual_loss(self, t: Tensor) -> Tensor:
        """Returns the residual loss evaluated at a batch of time stamps

        Args:
            t (Tensor): batch of time stamps

        Returns:
            Tensor: residual evaulated at time stamps t
        """
        return self.marginal_sde(t, self.sde_prior.drift, self.n_reparam_samples)

    def combine_elbo_terms(
        self, beta, N, log_like, residual_loss, kl_divergence, compat_mode=True
    ):
        if compat_mode:
            return log_like - 0.5 / N * residual_loss - beta / N * kl_divergence

        return log_like - (0.5 * residual_loss + kl_divergence) * beta / N

    def elbo(
        self,
        t: Tensor,
        x: Tensor,
        beta: float,
        N: int,
        print_loss=False,
        compat_mode=True,
    ) -> Tensor:
        """
        Computes the reweighted evidence lower bound (i.e. ELBO / N)

        Args:
            t (Tensor): (bs,) time stamps of measurements
            x (Tensor): (bs, D) measurements
            beta (float): weight of kl-divergence
            N (int): total number of data points

        Returns:
            Tensor: normalized evidence lower bound
        """
        # todo: double check, but I think that I should resample the sde and diffusion priors here only
        self.sde_prior.resample_weights()
        self.diffusion_prior.resample_weights()
        log_like = self.likelihood.mean_log_likelihood(
            t, x, self.marginal_sde, self.n_reparam_samples
        )
        if beta == 0:
            residual_loss = self.quad_scheme(self.residual_loss)
            kl_divergence = torch.tensor(0.0)
        else:
            residual_loss = self.quad_scheme(self.residual_loss)
            kl_divergence = (
                self.sde_prior.kl_divergence() + self.diffusion_prior.kl_divergence
            )
        if print_loss:
            print(
                f"log-likelihood: {log_like.item():.2f}, log-residual-likelihood: {-residual_loss*0.5/N:.2f}, kl-divergence: {kl_divergence.item():.2f}"
            )
        # loss = log_like - 0.5 / N * residual_loss - beta / N * kl_divergence
        loss = self.combine_elbo_terms(
            beta, N, log_like, residual_loss, kl_divergence, compat_mode=compat_mode
        )
        return loss

    # methods for making predictions
    def drift(self, t: Tensor, z: Tensor, summary_type: str = "sample") -> Tensor:
        """Returns the drift function for the approximate SDE

        Args:
            t (Tensor): time stamps to evaluate drift function
            z (Tensor): states at time stamps

        Returns:
            Tensor: drift evaluated at time stamps and states
        """
        if summary_type == "mean":
            return self.sde_prior.drift(t, z).mean(0)
        elif summary_type == "sample":
            return self.sde_prior.drift(t, z, integration_mode=True)
        else:
            raise ValueError(f"summary_type {summary_type} not recognized.")

    def diffusion(self, summary_type: str = "sample") -> Tensor:
        """Returns the diffusion matrix

        Args:
            summary_type (str, optional): "sample" or "mean"

        Returns:
            Tensor: the diffusion matrix
        """
        if summary_type == "mean":
            return torch.linalg.cholesky(self.diffusion_prior.process_noise.mean(0))
        elif summary_type == "sample":
            return torch.linalg.cholesky(self.diffusion_prior.process_noise)
        else:
            raise ValueError(f"summary_type {summary_type} not recognized.")

    def resample_sde_params(self, n: int = None) -> None:
        """Resample the sde_prior and diffusion_prior

        Args:
            n (int, optional): number of samples to use when resampling
        """
        if n is not None:
            n_reparam_samples = self.sde_prior.n_reparam_samples
            self.sde_prior.n_reparam_samples = n
            self.diffusion_prior.n_reparam_samples = n
        self.sde_prior.resample_weights()
        self.diffusion_prior.resample_weights()
        # reset n_reparam_samples
        if n is not None:
            self.sde_prior.n_reparam_samples = n_reparam_samples
            self.diffusion_prior.n_reparam_samples = n_reparam_samples


class SparsePolynomialSDE(SDELearner):
    """Subclass of SDELearner for the case that the prior over the
    drift function is a sparse linear combination of polynomials, the
    observation matrix is linear, observation noise is a diagonal
    Gaussian, and the Markov GP is parameterized by RBF models
    with the Matern 5/2 kernel. Also performs some useful
    initialization to make training more stable.

    Args:
        d (int): dimension of the state
        t_span (Union[Tuple[float, float], Tuple[Tensor, Tensor]]): min and max boundary for RBF centers
        degree (int): degree of polynomial in drift function
        n_reparam_samples (int): number of samples to use when using the reparametrization trick
        G (Tensor): observation matrix (i.e. y = Gx)
        num_meas (int): number of observations (should be G.shape[0])
        measurement_noise (Tensor): variance of observations
        tau (float): global half-cauchy scaling parameter
        train_t (Tensor, optional): observation time stamps (ns,)
        train_x (Tensor, optional): observations (ns, num_meas)
        input_labels (List[str], optional): name of each variable (i.e. ["x", "y", ...])
        n_quad (int, optional): number of quadrature nodes to use
        quad_percent (float, optional): 1 - quad_percent of the quad_nodes will be sampled uniformly in the time window
        n_tau (int, optional): number of centers for the RBF models
    """

    def __init__(
        self,
        d: int,
        t_span: Union[Tuple[float, float], Tuple[Tensor, Tensor]],
        degree: int,
        n_reparam_samples: int,
        G: Tensor,
        num_meas: int,
        measurement_noise: Tensor,
        tau: float = 1e-5,
        train_t: Tensor = None,
        train_x: Tensor = None,
        input_labels: List[str] = None,
        n_quad: int = 128,
        quad_percent: float = 0.8,
        n_tau: int = 200,
    ) -> None:
        assert quad_percent > 0 and quad_percent < 1, "quad_percent must be in (0, 1)"
        Q_diag = torch.ones(d) * 1e-0
        diffusion_prior = SparseDiagonalDiffusionPrior(
            d, Q_diag, n_reparam_samples, tau
        )
        fast_initialization = (
            (torch.linalg.matrix_rank(G) == d)
            and (train_t is not None)
            and (train_x is not None)
        )
        # transform = IdentityScaleTransform(d)
        if fast_initialization:
            stdev_transform = (torch.eye(d) - G).sum() == 0 and (train_x is not None)
            if stdev_transform:
                # transform = StdevScaleTransform(train_x)
                measurement_noise = measurement_noise  # * (1 / transform.scale ** 2)
            # train_x = transform(train_x)
            z = solve_least_squares(G, train_x.t(), gamma=1e-2).t()
            # , clamp_min=1e-1).t()
        else:
            train_t = None
            z = None
        marginal_sde = SpectralMarginalSDE(
            d,
            t_span,
            diffusion_prior=diffusion_prior,
            model_form="GLM",
            n_tau=n_tau,
            learn_inducing_locations=False,
            train_x=train_t,
            train_y=z,
        )
        quad_scheme = UnbiasedGaussLegendreQuad(
            t_span[0], t_span[1], n_quad, quad_percent=quad_percent
        )
        likelihood = IndepGaussLikelihood(G, num_meas, measurement_noise)
        features = SparsePolynomialFeatures(d, degree=degree, input_labels=input_labels)
        if fast_initialization:
            m, dmdt = marginal_sde.mean(train_t, return_grad=True)
        else:
            m, dmdt = (None, None)
        sde_prior = SparseMultioutputGLM(
            d,
            SparseFeatures=features,
            n_reparam_samples=n_reparam_samples,
            tau=tau,
            train_x=m,
            train_y=dmdt,
            # transform=transform,
        )
        super().__init__(
            marginal_sde,
            likelihood,
            quad_scheme,
            sde_prior,
            diffusion_prior,
            n_reparam_samples,
        )
        # violates Liskov substitution principle...
        # self.transform = transform

    def elbo(
        self, t: Tensor, x: Tensor, beta: float, N: int, print_loss=False
    ) -> Tensor:
        return super().elbo(t, x, beta, N, print_loss=print_loss)


class SparsePolynomialIntegratorSDE(SDELearner):
    """
    Assumes that states are governed by a drift function which is a function
    of both the position and it's velocity but we can only observe
    the states.

    Like the SparsePolynomialSDE, this is a
    Subclass of SDELearner for the case that the prior over the
    drift function is a sparse linear combination of polynomials, the
    observation matrix is linear, observation noise is a diagonal
    Gaussian, the Markov GP is parameterized by RBF models
    with the Matern 5/2 kernel. Also performs some useful
    initialization to make training more stable.

    Args:
        d (int): dimension of the state
        t_span (Union[Tuple[float, float], Tuple[Tensor, Tensor]]): min and max boundary for RBF centers
        degree (int): degree of polynomial in drift function
        n_reparam_samples (int): number of samples to use when using the reparametrization trick
        G (Tensor): observation matrix (i.e. y = Gx)
        num_meas (int): number of observations (should be G.shape[0])
        measurement_noise (Tensor): variance of observations
        tau (float): global half-cauchy scaling parameter
        train_t (Tensor, optional): observation time stamps (ns,)
        train_x (Tensor, optional): observations (ns, num_meas)
        input_labels (List[str], optional): name of each variable (i.e. ["x", "y", ...])
        n_tau (int, optional): number of centers for the RBF models
    """

    def __init__(
        self,
        d: int,
        t_span: Union[Tuple[float, float], Tuple[Tensor, Tensor]],
        degree: int,
        n_reparam_samples: int,
        G: Tensor,
        num_meas: int,
        measurement_noise: Tensor,
        tau: float = 1e-5,
        train_t: Tensor = None,
        train_x: Tensor = None,
        input_labels: List[str] = None,
        n_tau: int = 200,
    ) -> None:
        Q_diag = torch.ones(d) * 1e-0
        diffusion_prior = SparseDiagonalDiffusionPrior(
            d, Q_diag, n_reparam_samples, tau
        )
        fast_initialization = (
            # (torch.linalg.matrix_rank(G) == d)
            (torch.eye(d)[: d // 2] - G).pow(2).sum() == 0
            and (train_t is not None)
            and (train_x is not None)
        )
        if fast_initialization:
            z = solve_least_squares(G, train_x.t(), gamma=1e-2).t()
        else:
            train_t = None
            z = None
        marginal_sde = SpectralMarginalSDE(
            d,
            t_span,
            diffusion_prior=diffusion_prior,
            model_form="GLM",
            n_tau=n_tau,
            learn_inducing_locations=False,
            train_x=train_t,
            train_y=z,
        )
        if fast_initialization:
            m, dmdt = marginal_sde.mean(train_t, return_grad=True)
            G_tmp = torch.cat([G, torch.eye(d)[d // 2 :]], dim=0)
            z = solve_least_squares(
                G_tmp,
                torch.cat([train_x, dmdt[:, : d // 2]], dim=-1).t(),
                gamma=1e-2,
            ).t()
        marginal_sde = SpectralMarginalSDE(
            d,
            t_span,
            diffusion_prior=diffusion_prior,
            model_form="GLM",
            n_tau=200,
            learn_inducing_locations=False,
            train_x=train_t,
            train_y=z,
        )
        quad_scheme = UnbiasedGaussLegendreQuad(
            t_span[0], t_span[1], 128, quad_percent=0.8
        )
        likelihood = IndepGaussLikelihood(G, num_meas, measurement_noise)
        features = SparsePolynomialFeatures(d, degree=degree, input_labels=input_labels)
        if fast_initialization:
            m, dmdt = marginal_sde.mean(train_t, return_grad=True)
        else:
            m, dmdt = (None, None)
        sde_prior = SparseIntegratorGLM(
            d,
            SparseFeatures=features,
            n_reparam_samples=n_reparam_samples,
            tau=tau,
            train_x=m,
            train_y=dmdt,
        )
        super().__init__(
            marginal_sde,
            likelihood,
            quad_scheme,
            sde_prior,
            diffusion_prior,
            n_reparam_samples,
        )


class StateEstimator(SDELearner):
    """
    Subclass of SDELearner for the case the drift and diagional diffusion matrix is known,
    the observation matrix is linear, observation noise is a diagonal
    Gaussian, the Markov GP is parametrized by RBF models
    with the Matern 5/2 kernel. Also performs some useful
    initialization to make training more stable.

    Args:
        d (int): dimension of the state
        t_span (Union[Tuple[float, float], Tuple[Tensor, Tensor]]): min and max boundary for RBF centers
        n_reparam_samples (int): number of samples to use when using the reparametrization trick
        G (Tensor): observation matrix (i.e. y = Gx)
        drift (Callable[[Tensor, Tensor], Tensor]): known drift function
        num_meas (int): number of observations (should be G.shape[0])
        measurement_noise (Tensor): variance of observations
        tau (float): global half-cauchy scaling parameter
        train_t (Tensor, optional): observation time stamps (ns,)
        train_x (Tensor, optional): observations (ns, num_meas)
        input_labels (List[str], optional): name of each variable (i.e. ["x", "y", ...])
        n_tau (int, optional): number of centers for the RBF models
        Q_diag (Tensor, optional): diagonal component of the diffusion matrix,
    """

    def __init__(
        self,
        d: int,
        t_span: Union[Tuple[float, float], Tuple[Tensor, Tensor]],
        n_reparam_samples: int,
        G: Tensor,
        drift: Callable[[Tensor, Tensor], Tensor],
        num_meas: int,
        measurement_noise: Tensor,
        tau: float = 1e-5,
        train_t: Tensor = None,
        train_x: Tensor = None,
        n_quad: int = 128,
        quad_percent: float = 0.8,
        n_tau: int = 200,
        Q_diag: Tensor = None,
    ) -> None:
        assert quad_percent > 0 and quad_percent < 1, "quad_percent must be in (0, 1)"
        if Q_diag is None:
            Q_diag = torch.ones(d) * 1e-0
            diffusion_prior = SparseDiagonalDiffusionPrior(
                d, Q_diag, n_reparam_samples, tau
            )
        else:
            diffusion_prior = ConstantDiagonalDiffusionPrior(d, Q_diag)
        fast_initialization = (
            (torch.linalg.matrix_rank(G) == d)
            and (train_t is not None)
            and (train_x is not None)
        )
        # transform = IdentityScaleTransform(d)
        if fast_initialization:
            stdev_transform = (torch.eye(d) - G).sum() == 0 and (train_x is not None)
            if stdev_transform:
                # transform = StdevScaleTransform(train_x)
                measurement_noise = measurement_noise  # * (1 / transform.scale ** 2)
            # train_x = transform(train_x)
            z = solve_least_squares(G, train_x.t(), gamma=1e-2).t()
            # , clamp_min=1e-1).t()
        else:
            train_t = None
            z = None
        marginal_sde = SpectralMarginalSDE(
            d,
            t_span,
            diffusion_prior=diffusion_prior,
            model_form="GLM",
            n_tau=n_tau,
            learn_inducing_locations=False,
            train_x=train_t,
            train_y=z,
        )
        quad_scheme = UnbiasedGaussLegendreQuad(
            t_span[0], t_span[1], n_quad, quad_percent=quad_percent
        )
        likelihood = IndepGaussLikelihood(G, num_meas, measurement_noise)
        if fast_initialization:
            m, dmdt = marginal_sde.mean(train_t, return_grad=True)
        else:
            m, dmdt = (None, None)
        sde_prior = ExactMotionModel(drift)
        super().__init__(
            marginal_sde,
            likelihood,
            quad_scheme,
            sde_prior,
            diffusion_prior,
            n_reparam_samples,
        )

    def elbo(
        self, t: Tensor, x: Tensor, beta: float, N: int, print_loss=False
    ) -> Tensor:
        return super().elbo(t, x, beta, N, print_loss=print_loss)


class NeuralSDE(SDELearner):
    """
    Subclass of SDELearner when the drift function is represented by a fully connected
    neural network.

    Args:
        d (int): dimension of the state
        t_span (Union[Tuple[float, float], Tuple[Tensor, Tensor]]): min and max boundary for RBF centers
        n_reparam_samples (int): number of samples to use when using the reparametrization trick
        G (Tensor): observation matrix (i.e. y = Gx)
        drift_layer_description (List[int]): number of neurons in each layer of the drift function
        nonlinearity (nn.Module): nonlinearity to use in the drift function
        measurement_noise (Tensor): variance of observations
        tau (float): global half-cauchy scaling parameter
        train_t (Tensor, optional): observation time stamps (ns,)
        train_x (Tensor, optional): observations (ns, num_meas)
        n_quad (int, optional): number of quadrature nodes to use
        quad_percent (float, optional): 1 - quad_percent of the quad_nodes will be sampled uniformly in the time window
        n_tau (int, optional): number of centers for the RBF models
        Q_diag (Tensor, optional): diagonal component of the diffusion matrix
    """

    def __init__(
        self,
        d: int,
        t_span: Union[Tuple[float, float], Tuple[Tensor, Tensor]],
        n_reparam_samples: int,
        G: Tensor,
        drift_layer_description: List[int],
        nonlinearity: nn.Module,
        measurement_noise: Tensor,
        tau: float = 1e-5,
        train_t: Optional[Tensor] = None,
        train_x: Optional[Tensor] = None,
        n_quad: int = 128,
        quad_percent: float = 0.8,
        n_tau: int = 200,
        Q_diag: Optional[Tensor] = None,
    ) -> None:
        assert quad_percent > 0 and quad_percent < 1, "quad_percent must be in (0, 1)"
        if Q_diag is None:
            Q_diag = torch.ones(d) * 1e-0
            diffusion_prior = SparseDiagonalDiffusionPrior(
                d, Q_diag, n_reparam_samples, tau
            )
        else:
            diffusion_prior = ConstantDiagonalDiffusionPrior(d, Q_diag)
        fast_initialization = (
            (torch.linalg.matrix_rank(G) == d)
            and (train_t is not None)
            and (train_x is not None)
        )
        if fast_initialization:
            assert isinstance(train_x, Tensor)  # helping the type checker
            z = solve_least_squares(G, train_x.t(), gamma=1e-2).t()
        else:
            train_t = None
            z = None
        marginal_sde = DiagonalMarginalSDE(
            d,
            t_span,
            diffusion_prior=diffusion_prior,
            model_form="GLM",
            n_tau=n_tau,
            learn_inducing_locations=False,
            train_x=train_t,
            train_y=z,
        )
        quad_scheme = UnbiasedGaussLegendreQuad(
            t_span[0], t_span[1], n_quad, quad_percent=quad_percent
        )
        likelihood = IndepGaussLikelihood(G, G.shape[0], measurement_noise)
        sde_prior = DriftFCNN(d, drift_layer_description, nonlinearity)
        super().__init__(
            marginal_sde,
            likelihood,
            quad_scheme,
            sde_prior,
            diffusion_prior,
            n_reparam_samples,
        )
