import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor

# from functorch import vmap, jacfwd
import math
from typing import Union, Tuple, Callable, Iterator, List, Optional
from abc import ABC, abstractmethod
from ..utils import *
from ..kernels import (
    Matern52,
    Matern12,
    Matern52withGradients,
    KumaraswamyWarping,
    IdentityWarp,
)
from ._diffusion_prior import *
import warnings
import numpy as np

__all__ = [
    "MarginalModel",
    "MarginalFCNN",
    "MeanFCNN",
    "StrictlyPositiveFCNN",
    "OrthogonalFCNN",
    "MarginalGLM",
    "MeanGLM",
    "StrictlyPositiveGLM",
    "OrthogonalGLM",
    "MarginalSDE",
    "DiagonalMarginalSDE",
    "SpectralMarginalSDE",
]


class MarginalModel(ABC, nn.Module):
    """
    Abstract base class for a function that returns
    the marginal statistics of a Markov GP
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self, t: Tensor, return_grad=False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Returns the either the output or output + gradient
        at the time stamps, t
            self ([TODO:parameter]): [TODO:description]
            return_grad (bool, optional ): flag indicating whether to just output or output and grad
            t (Tensor): batch of time stamps

        Returns:
            output or output and grads
        """
        pass


class MarginalFCNN(MarginalModel):
    """
    DEPRECATED
    """

    def __init__(
        self,
        num_outputs,
        t_span,
        num_hidden,
        num_layers,
        output_transform: Callable = Identity(),
    ):
        super().__init__()
        batch_norm = False
        nonlinearity = nn.SiLU()
        self.fcnn = FCNN(
            1,
            num_hidden,
            num_outputs,
            num_layers,
            nonlinearity=nonlinearity,
            output_transform=output_transform,
            batch_norm=batch_norm,
        )
        if not isinstance(t_span[0], Tensor):
            dt = torch.tensor(t_span[1] - t_span[0])
        else:
            dt = (t_span[1] - t_span[0]).detach()
        self.register_buffer("dt", dt)

    def forward(
        self, t: Tensor, return_grad=False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        # normalize inputs (assumes t is uniformly sampled)
        t = t.mul(2.0 / self.dt).add(-1.0).mul(math.sqrt(3))
        if t.dim() == 0:
            t = t.unsqueeze(-1)
        # turn on gradient tracking
        flip_grad = False
        if not t.requires_grad and return_grad:
            t.requires_grad_(True)
            flip_grad = True
        m = self.fcnn(t)
        if not return_grad:
            return m
        else:
            dmdt = (
                multioutput_gradient(m, t, vmap=False)
                .squeeze(-2)
                .mul(2.0 / self.dt * math.sqrt(3))
            )
            if flip_grad:
                t.requires_grad_(False)
            return (m, dmdt)


class MeanFCNN(MarginalFCNN):
    def __init__(
        self,
        num_outputs,
        t_span,
        num_hidden,
        num_layers,
        train_t: Tensor = None,
        train_x: Tensor = None,
    ):
        super().__init__(num_outputs, t_span, num_hidden, num_layers)
        if train_t is not None:
            assert train_x is not None, "train_x and train_y must both be provided"
            t, y = (train_t, train_x)
            self.train_fcnn(t, y)
        elif train_x is not None:
            raise ValueError("train_x and train_y must both be provided.")

    def forward(
        self, t: Tensor, return_grad=False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        return super().forward(t, return_grad=return_grad)

    def train_fcnn(self, train_t, train_x):
        train_dataset = TensorDataset(train_t, train_x)
        num_mc_samples = 128
        train_loader = DataLoader(
            train_dataset, batch_size=num_mc_samples, shuffle=True
        )
        # num_epochs = num_iters // len(train_loader)
        num_epochs = 2000
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        for j in range(num_epochs):
            for t_batch, y_batch in train_loader:
                optimizer.zero_grad()
                y_pred = self.forward(t_batch)
                (y_pred - y_batch).pow(2).sum(-1).mean().backward()
                optimizer.step()


class StrictlyPositiveFCNN(MarginalFCNN):
    def __init__(self, num_outputs, t_span, num_hidden, num_layers):
        super().__init__(
            num_outputs, t_span, num_hidden, num_layers, output_transform=Positive()
        )
        self.fcnn.mlp[-2].bias.data.fill_(1.0)

    def forward(
        self, t: Tensor, return_grad=False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if return_grad:
            m, dmdt = super().forward(t, return_grad=True)
            return (m, dmdt)
        else:
            return super().forward(t, return_grad=False)


class OrthogonalFCNN(MarginalFCNN):
    def __init__(self, num_outputs, t_span, num_hidden, num_layers):
        num_skew = num_outputs * (num_outputs - 1) // 2
        super().__init__(num_skew, t_span, num_hidden, num_layers)
        self.orthogonal_mexp = SkewSymMatrixExp(num_outputs)
        self.num_outputs = num_outputs

    def forward(
        self, t: Tensor, return_grad=False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if not return_grad:
            m = super().forward(t, return_grad=False)
            return self.orthogonal_mexp(m, return_grad=False)
        else:
            m, dmdt = super().forward(t, return_grad=True)
        expS, grad_expS = self.orthogonal_mexp(m, return_grad=True)
        dexpSdt = (grad_expS @ (dmdt).unsqueeze(-1)).squeeze(-1)
        return (
            expS,
            dexpSdt.view(len(t), self.num_outputs, self.num_outputs).transpose(-2, -1),
        )


class MarginalGLM(MarginalModel):
    def __init__(
        self,
        num_outputs,
        t_span,
        n_tau,
        learn_inducing_locations: bool = False,
        whitened_param: bool = True,
        kernel: str = "matern52",
        apply_input_warping: bool = True,
        len_init: float = 1.0,
        dynamic_update_bounds: bool = True,
    ) -> None:
        super().__init__()
        nu = 1.0
        tau = torch.linspace(t_span[0] - nu, t_span[1] + nu, n_tau).unsqueeze(-1)
        if apply_input_warping:
            input_warping = KumaraswamyWarping(
                (t_span[0] - nu, t_span[1] + nu),
                dynamic_update_bounds=dynamic_update_bounds,
            )
        else:
            input_warping = IdentityWarp()
        self.learn_inducing_locations = learn_inducing_locations
        self.whitened_param = whitened_param
        if learn_inducing_locations:
            self.tau = Parameter(tau)
        else:
            self.register_buffer("tau", tau)
        # if init_with_grad = True, K(tau, tau) will be size (n_tau, 2*n_tau)
        init_with_grad = False
        if kernel == "matern52":
            self.K = Matern52(input_warping=input_warping, len_init=len_init)
        elif kernel == "matern52withgradients":
            init_with_grad = True
            self.K = Matern52withGradients(
                input_warping=input_warping, len_init=len_init
            )
        elif kernel == "matern12":
            self.K = Matern12(input_warping=input_warping, len_init=len_init)
        else:
            raise ValueError(f"Unknown kernel {kernel}")
        self.register_buffer("eps", torch.tensor(1e-6))
        if not init_with_grad:
            K_tmp = self.K(self.tau, self.tau)
        else:
            warnings.warn(
                "Learning inducing locations is not supported with gradient glms."
            )
            K_tuple = self.K(self.tau, self.tau, return_grad=True)
            K_tmp = torch.cat(K_tuple, dim=0)
        n_weights = K_tmp.shape[-1]
        C = torch.linalg.cholesky(K_tmp + self.eps * torch.eye(n_weights))
        self.register_buffer("C_buffer", C.detach())
        self.raw_w = Parameter(torch.Tensor(n_weights, num_outputs))
        self.b = Parameter(torch.Tensor(num_outputs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.whitened_param:
            nn.init.normal_(self.raw_w, mean=0.0, std=1.0)
        else:
            w_init = torch.triangular_solve(self.raw_w, self.C.t()).solution
            self.raw_w.data.copy_(w_init)
        nn.init.zeros_(self.b)

    @property
    def n_tau(self) -> int:
        return len(self.tau)

    @property
    def C(self):
        if self.learn_inducing_locations:
            C_out = torch.linalg.cholesky(
                self.K(self.tau, self.tau) + self.eps * torch.eye(self.n_tau)
            )
        else:
            C_out = self.C_buffer
        return C_out

    @property
    def w(self):
        if self.whitened_param:
            return torch.triangular_solve(self.raw_w, self.C.t()).solution
        else:
            return self.raw_w

    @w.setter
    def w(self, value: Tensor):
        if self.whitened_param:
            self.raw_w.data.copy_(self.C.t() @ value)
        else:
            self.raw_w.data.copy_(value)

    def forward(
        self, t: Tensor, return_grad=False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if t.dim() == 0:
            t = t.unsqueeze(-1)
        if not return_grad:
            return self.K(t, self.tau) @ self.w + self.b
        else:
            m, dmdt = self.K(t, self.tau, return_grad=True)
            return m @ self.w + self.b, dmdt @ self.w


def train_glm(
    train_x,
    train_y,
    train_dydx,
    num_outputs,
    t_span,
    n_tau,
    whitened_param,
    learn_inducing_locations,
    kernel,
):
    len_init_list = [1e-1, 0.5, 1.0, 10.0]
    eval_error = {len_init: [] for len_init in len_init_list}
    n_splits = 5
    idx = np.arange(len(train_x))
    np.random.shuffle(idx)
    n = len(train_x) // n_splits
    split_idx = [idx[n * i : n * (i + 1)] for i in range(n_splits)]
    for len_init in len_init_list:
        glm = MarginalGLM(
            num_outputs,
            t_span,
            n_tau,
            whitened_param=whitened_param,
            learn_inducing_locations=learn_inducing_locations,
            kernel=kernel,
            len_init=len_init,
        )
        for split in split_idx:
            train_idx = np.array(list(set(idx) - set(split)))
            t = train_x[train_idx]
            y = train_y[train_idx]
            t_eval = train_x[split]
            y_eval = train_y[split]
            if train_dydx is not None:
                y = torch.cat([y, train_dydx[train_idx]], dim=0)
                y_eval = torch.cat([y_eval, train_dydx[split]], dim=0)
                Kttau, dKttaudt = glm.K(t, glm.tau, return_grad=True)
                train_features = torch.cat([Kttau, dKttaudt], dim=0)
                Kvttau, dKvttaudt = glm.K(t_eval, glm.tau, return_grad=True)
                val_features = torch.cat([Kvttau, dKvttaudt], dim=0)
            else:
                train_features = glm.K(t, glm.tau)
                val_features = glm.K(t_eval, glm.tau)
            w = solve_least_squares(train_features, y, gamma=1e-1)  # , clamp_min=1e-6)
            eval_loss = (val_features @ w - y_eval).pow(2).mean().detach().numpy()
            eval_error[len_init].append(eval_loss)
        eval_error[len_init] = np.array(eval_error[len_init]).mean()
    best_len = len_init_list[0]
    eval_best = eval_error[best_len]
    for len_init in len_init_list:
        if eval_error[len_init] < eval_best:
            best_len = len_init
            eval_best = eval_error[len_init]
    return best_len


class MeanGLM(MarginalGLM):
    def __init__(
        self,
        num_outputs,
        t_span,
        n_tau,
        learn_inducing_locations: bool = False,
        whitened_param: bool = True,
        kernel: str = "matern52",
        train_x: Tensor = None,
        train_y: Tensor = None,
        train_dydx: Tensor = None,
    ) -> None:
        if train_x is not None:
            assert train_y is not None, "train_x and train_y must both be provided"
            # determinte a good initialization for the length scale
            len_init = train_glm(
                train_x,
                train_y,
                train_dydx,
                num_outputs,
                t_span,
                n_tau,
                whitened_param,
                learn_inducing_locations,
                kernel,
            )
        else:
            len_init = 1.0
        super().__init__(
            num_outputs,
            t_span,
            n_tau,
            whitened_param=whitened_param,
            learn_inducing_locations=learn_inducing_locations,
            kernel=kernel,
            len_init=len_init,
        )
        if train_x is not None:
            assert train_y is not None, "train_x and train_y must both be provided"
            t, y = (train_x, train_y)
            if train_dydx is not None:
                y = torch.cat([y, train_dydx], dim=0)
                K, dKdt = self.K(t, self.tau, return_grad=True)
                features = torch.cat([K, dKdt], dim=0)
            else:
                features = self.K(t, self.tau)
            # w = solve_least_squares(features, y, gamma=1e-1, clamp_min=1e-6)
            w = solve_least_squares(features, y, gamma=1e-1)  # , clamp_min=1e-32)
            self.w = w
            self.b.data.copy_(torch.zeros_like(self.b.data))
        elif train_y is not None:
            raise ValueError("train_x and train_y must both be provided.")

    def forward(
        self, t: Tensor, return_grad=False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if t.dim() == 0:
            t = t.unsqueeze(-1)
        if not return_grad:
            return self.K(t, self.tau) @ self.w + self.b
        else:
            m, dmdt = self.K(t, self.tau, return_grad=True)
            return m @ self.w + self.b, dmdt @ self.w


class StrictlyPositiveGLM(MarginalGLM):
    def __init__(
        self,
        num_outputs,
        t_span,
        n_tau,
        learn_inducing_locations: bool = False,
        kernel: str = "matern52",
    ) -> None:
        super().__init__(
            num_outputs,
            t_span,
            n_tau,
            learn_inducing_locations=learn_inducing_locations,
            kernel=kernel,
        )
        self.output_transform = Positive()
        nn.init.constant_(self.b, -2.2522)
        self.w = torch.zeros(self.n_tau, num_outputs)

    def forward(
        self, t: Tensor, return_grad=False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if t.dim() == 0:
            t = t.unsqueeze(-1)
        if not return_grad:
            return self.output_transform(self.K(t, self.tau) @ self.w + self.b)
        else:
            m, dmdt = self.K(t, self.tau, return_grad=True)
            res, grad_res = self.output_transform(m @ self.w + self.b, return_grad=True)
            return res, grad_res * (dmdt @ self.w)


class OrthogonalGLM(MarginalGLM):
    def __init__(
        self,
        num_outputs,
        t_span,
        n_tau,
        learn_inducing_locations: bool = False,
        kernel: str = "matern52",
    ) -> None:
        super().__init__(
            num_outputs * (num_outputs - 1) // 2,
            t_span,
            n_tau,
            learn_inducing_locations=learn_inducing_locations,
            kernel=kernel,
        )
        self.orthogonal_mexp = SkewSymMatrixExp(num_outputs)
        self.num_outputs = num_outputs
        nn.init.constant_(self.w, 1e-6)

    def forward(
        self, t: Tensor, return_grad=False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if t.dim() == 0:
            t = t.unsqueeze(-1)
        if not return_grad:
            return self.orthogonal_mexp(self.K(t, self.tau) @ self.w + self.b)
        else:
            m, dmdt = self.K(t, self.tau, return_grad=True)
            expS, grad_expS = self.orthogonal_mexp(
                m @ self.w + self.b, return_grad=True
            )
            # todo: grad_expS.mul(dmdt @ self.w)?
            dexpSdt = (grad_expS @ (dmdt @ self.w).unsqueeze(-1)).squeeze(-1)
            return (
                expS,
                dexpSdt.view(len(t), self.num_outputs, self.num_outputs).transpose(
                    -2, -1
                ),
            )


class MarginalSDE(ABC, nn.Module):
    """
    Abstract base class for a model of the marginal statistics of
    a Markov Gaussian process.
    """

    def __init__(self) -> None:
        super().__init__()
        self.device_var = Parameter(torch.empty(0))
        self.low_rank_cov = False  # bool indicating if covariance is low rank

    @property
    def device(self):
        return self.device_var.device

    @abstractmethod
    def mean_parameters(self) -> Iterator[Parameter]:
        """
        Returns an iterator over the mean parameters of the marginal SDE.

        Returns:
            Iterator[Parameter]: An iterator over the mean parameters of the marginal SDE.
        """
        pass

    @abstractmethod
    def generate_samples(self, t: Tensor, num_samples: int, *args, **kwargs) -> Tensor:
        """
        Generates samples from the approximating SDE marginal distribution at time t and optionally
        returns some intermediate quantities.

        Args:
            t (Tensor): (bs, ) time stamps at which to generate samples
            num_samples (int): number of independent samples at each time stamp

        Returns:
            Tensor: samples of latent states
        """
        pass

    @abstractmethod
    def forward(
        self, t: Tensor, f: Callable[[Tensor, Tensor], Tensor], num_samples: int
    ) -> Tensor:
        """
        Computes the (unweighted) residual loss between the approximating and prior SDEs.

        Args:
            t (Tensor): (bs, ) time stamps at which to generate samples
            f (Callable[[Tensor,Tensor],Tensor]): (t, z) -> (num_samples, bs, d) or (bs, d) prior SDE evaluations
            num_samples (int): number of reparameterized samples at each time stamp

        Returns:
            Tensor: (bs,) approximate residual loss
        """
        pass

    @abstractmethod
    def K(self, t: Tensor) -> Tensor:
        """
        Returns the the covariance matrix K(t) evaluated at a batch of times.

        Args:
            t (Tensor): Time stamps at which to compute the covariance matrix

        Returns:
            Tensor: (len(b), d, d) batch of covariance matrices
        """
        pass

    @abstractmethod
    def mean(self, t: Tensor, return_grad=False) -> Tensor:
        """
        Returns the marginal mean at time t

        Args:
            t (Tensor): (bs,) time stamps at which to evaluate mean

        Returns:
            Tensor: (bs, d) mean evaluated at times
        """
        pass


class DiagonalMarginalSDE(MarginalSDE):
    """
    A model for the marginal statistics of a Markov GP whose
    covariance is diagonal.

    Args:
        d (int): dimension of the state
        t_span (Union[Tuple[float, float], Tuple[Tensor, Tensor]]): time span of data
        diffusion_prior (DiagonalDiffusionPrior): Some diagonal diffusion prior
        model_form (str, optional): GLM or FCNN (FCNN not fully tested)
        vmap (bool, optional): DEPRECATED
        **kwargs: arguments passed onto GLM / FCNN
    """

    def __init__(
        self,
        d: int,
        t_span: Union[Tuple[float, float], Tuple[Tensor, Tensor]],
        diffusion_prior: DiagonalDiffusionPrior,
        model_form: str = "GLM",
        vmap: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.vmap = vmap
        t0 = t_span[0]
        tf = t_span[1]
        self.d = d
        self.diffusion_prior = diffusion_prior
        train_x = kwargs.get("train_x", None)
        train_y = kwargs.get("train_y", None)
        train_dydx = kwargs.get("train_dydx", None)
        if model_form == "GLM":
            kernel = kwargs.get("kernel", "matern52")
            n_tau = kwargs.get("n_tau", 100)
            learn_inducing_locations = kwargs.get("learn_inducing_locations", False)
            self.m = MeanGLM(
                d,
                (t0, tf),
                n_tau,
                learn_inducing_locations=learn_inducing_locations,
                kernel=kernel,
                train_x=train_x,
                train_y=train_y,
                train_dydx=train_dydx,
            )
            kernel = "matern52"
            self.K_diag = StrictlyPositiveGLM(
                d,
                (t0, tf),
                n_tau,
                learn_inducing_locations=learn_inducing_locations,
                kernel=kernel,
            )
        elif model_form == "FCNN":
            hidden_size = kwargs.get("hidden_size", 50)
            num_layers = kwargs.get("num_layers", 2)
            self.m = MeanFCNN(
                d, (t0, tf), hidden_size, num_layers, train_t=train_x, train_x=train_y
            )
            self.K_diag = StrictlyPositiveFCNN(d, (t0, tf), hidden_size, num_layers)
        else:
            raise ValueError(f"Invalid model_form: {model_form}")

    def mean(self, t: Tensor, return_grad=False) -> Tensor:
        """
        Returns the marginal mean at time t

        Args:
            t (Tensor): (bs,) time stamps at which to evaluate mean

        Returns:
            Tensor: (bs, d) mean evaluated at times
        """
        if return_grad:
            return self.m(t, return_grad=True)
        else:
            return self.m(t)

    def mean_parameters(self) -> Iterator[Parameter]:
        """
        Returns an iterator over the mean parameters of the marginal SDE.

        Returns:
            Iterator[Parameter]: An iterator over the mean parameters of the marginal SDE.
        """
        return self.m.parameters()

    def K(self, t: Tensor) -> Tensor:
        """
        Compute the marginal covariance matrix at time t.

        Args:
            t (Tensor): Time stamps at which to compute the covariance matrix

        Returns:
            Tensor: (len(b), d, d) batch of covariance matrices
        """
        return self.K_diag(t).diag_embed()

    def drift(self, t: Tensor, z: Tensor) -> Tensor:
        """
        compute the drift function of the equivalent SDE at times t.

        Args:
            t (Tensor): (bs,) time stamps at which to compute the drift
            z (Tensor): (..., bs, d) batch of latent states

        Returns:
            Tensor: (..., bs, d) batch of drift function evaluations
        """
        # computing parameterization and time derivative
        m, dmdt = self.m(t, return_grad=True)
        K_diag, dKdt_diag = self.K_diag(t, return_grad=True)
        p_noise = self.diffusion_prior.process_noise_diag
        # todo: check if this is correct for mle case
        if dKdt_diag.shape[0] != p_noise.shape[0] and p_noise.dim() == 2:
            p_noise = p_noise.unsqueeze(1)
        A = solve_lyapunov_diag(K_diag, -dKdt_diag + p_noise)
        return dmdt - (A * (z - m))

    def unweighted_residual_loss(self, drift: Tensor, f: Tensor) -> Tensor:
        """
        Computes the (unweighted) residual loss between the approximating and prior SDEs.

        Args:
            drift (Tensor): (n_samples, bs, d) drift function evaluations
            f (Tensor): (n_samples, bs, d) / (bs, d) prior SDE evaluations

        Returns:
            Tensor: residual loss
        """
        r = drift - f
        return self.diffusion_prior.solve_linear(r.pow(2)).sum(-1)

    def generate_samples(self, t: Tensor, num_samples: int) -> Tensor:
        """
        Generates samples from the approximating SDE marginal distribution at time t and optionally
        returns some intermediate quantities.

        Args:
            t (Tensor): (bs, ) time stamps at which to generate samples
            num_samples (int): number of independent samples at each time stamp
            return_intermediates (bool, optional): indicates whether to return intermediate quanties. Defaults to False.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]: samples of latent states or samples of latent states and intermediate quantities
        """

        v = torch.randn(num_samples, len(t), self.d, device=self.device)
        # get mean at each time stamp
        return self.mean(t) + (self.K_diag(t).sqrt().mul(v))

    def forward(
        self, t: Tensor, f: Callable[[Tensor, Tensor], Tensor], num_samples: int
    ) -> Tensor:
        """
        Computes the (unweighted) residual loss between the approximating and prior SDEs.

        Args:
            t (Tensor): (bs, ) time stamps at which to generate samples
            f (Callable[[Tensor,Tensor],Tensor]): (t, z) -> (num_samples, bs, d) or (bs, d) prior SDE evaluations
            num_samples (int): number of reparameterized samples at each time stamp

        Returns:
            Tensor: (bs,) approximate residual loss
        """
        # compute residual loss making good use of intermediate quantities
        zs = self.generate_samples(t, num_samples)
        drift = self.drift(t, zs)
        f_samples = f(t, zs)
        return self.unweighted_residual_loss(drift, f_samples).mean(0)


class SpectralMarginalSDE(MarginalSDE):
    """
    A model for the marginal statistics of a Markov GP using the
    spectral parametrization described in the main text.

    Args:
        d (int): dimension of the state
        t_span (Union[Tuple[float, float], Tuple[Tensor, Tensor]]): time span of data
        diffusion_prior (DiffusionPrior): Some diffusion prior
        model_form (str, optional): GLM or FCNN (FCNN not fully tested)
        vmap (bool, optional): DEPRECATED
        **kwargs: arguments passed onto GLM / FCNN
    """

    def __init__(
        self,
        d: int,
        t_span: Union[Tuple[float, float], Tuple[Tensor, Tensor]],
        diffusion_prior: DiffusionPrior,
        model_form: str = "GLM",
        vmap: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        t0 = t_span[0]
        tf = t_span[1]
        self.vmap = vmap
        self.d = d
        self.register_buffer("tril_ind", torch.tril_indices(d, d, offset=-1))
        self.diffusion_prior = diffusion_prior
        self.register_buffer("ident", torch.eye(d))
        train_x = kwargs.get("train_x", None)
        train_y = kwargs.get("train_y", None)
        if model_form == "GLM":
            kernel = kwargs.get("kernel", "matern52")
            n_tau = kwargs.get("n_tau", 100)
            learn_inducing_locations = kwargs.get("learn_inducing_locations", False)
            self.m = MeanGLM(
                d,
                (t0, tf),
                n_tau,
                kernel=kernel,
                learn_inducing_locations=learn_inducing_locations,
                train_x=train_x,
                train_y=train_y,
            )
            self.orthogonal = OrthogonalGLM(
                d,
                (t0, tf),
                n_tau,
                kernel=kernel,
                learn_inducing_locations=learn_inducing_locations,
            )
            self.eigenvals = StrictlyPositiveGLM(
                d,
                (t0, tf),
                n_tau,
                kernel=kernel,
                learn_inducing_locations=learn_inducing_locations,
            )
        elif model_form == "FCNN":
            hidden_size = kwargs.get("hidden_size", 50)
            num_layers = kwargs.get("num_layers", 2)
            # notation for train_t vs train_x is confusing
            self.m = MeanFCNN(
                d, (t0, tf), hidden_size, num_layers, train_t=train_x, train_x=train_y
            )
            self.orthogonal = OrthogonalFCNN(d, (t0, tf), hidden_size, num_layers)
            self.eigenvals = StrictlyPositiveFCNN(d, (t0, tf), hidden_size, num_layers)
        else:
            raise ValueError(f"Invalid model_form: {model_form}")

    def mean(self, t: Tensor, return_grad=False) -> Tensor:
        """
        Returns the marginal mean at time t

        Args:
            t (Tensor): (bs,) time stamps at which to evaluate mean

        Returns:
            Tensor: (bs, d) mean evaluated at times
        """
        if return_grad:
            return self.m(t, return_grad=True)
        else:
            return self.m(t)

    def mean_parameters(self) -> Iterator[Parameter]:
        """
        Returns an iterator over the mean parameters of the marginal SDE.

        Returns:
            Iterator[Parameter]: An iterator over the mean parameters of the marginal SDE.
        """
        return self.m.parameters()

    def K(self, t: Tensor) -> Tensor:
        """
        Compute the marginal covariance matrix at time t.

        Args:
            t (Tensor): Time stamps at which to compute the covariance matrix

        Returns:
            Tensor: (len(b), d, d) batch of covariance matrices
        """
        U = self.orthogonal(t)
        D = self.eigenvals(t)
        return (U.mul(D.unsqueeze(-2))) @ U.transpose(-2, -1)

    def _dKdt_fast(self, U: Tensor, dUdt: Tensor, D: Tensor, dDdt: Tensor) -> Tensor:
        """
        Compute the time derivative of the marginal covariance matrix using intermediate quantities.

        Args:
            U (Tensor): (bs, d, d) batch of orthogonal matrices
            dUdt (Tensor): (bs, d, d) batch of time derivatives of orthogonal matrices
            D (Tensor): (bs,d) batch of eigenvalues of the covariance matrix
            dDdt (Tensor): (bs,d) batch of time derivatives of eigenvalues of the covariance matrix

        Returns:
            Tensor: (bs, d, d) batch of time derivatives of the marginal covariance matrix
        """
        F1 = (U.mul(D.unsqueeze(-2))) @ dUdt.transpose(-2, -1)
        F2 = dUdt.mul(D.unsqueeze(-2))
        F3 = U.mul(dDdt.unsqueeze(-2))
        dKdt = F1 + (F2 + F3) @ U.transpose(-2, -1)
        return dKdt

    def drift(self, t: Tensor, z: Tensor) -> Tensor:
        """
        compute the drift function of the equivalent SDE at times t.

        Args:
            t (Tensor): (bs,) time stamps at which to compute the drift
            z (Tensor): (..., bs, d) batch of latent states

        Returns:
            Tensor: (..., bs, d) batch of drift function evaluations
        """
        # computing parameterization and time derivative
        U, dUdt = self.orthogonal(t, return_grad=True)
        D, dDdt = self.eigenvals(t, return_grad=True)
        return self._drift_fast(t, z, U, dUdt, D, dDdt)

    def _drift_fast(
        self, t: Tensor, z: Tensor, U: Tensor, dUdt: Tensor, D: Tensor, dDdt: Tensor
    ) -> Tensor:
        """
        Compute the drift function of the equivalent SDE at times t using intermediate quantities.

        Args:
            t (Tensor): (bs,) time stamps at which to compute the drift
            z (Tensor): (..., bs, d) batch of latent states
            U (Tensor): (bs, d, d) batch of orthogonal matrices
            dUdt (Tensor): (bs, d, d) batch of time derivatives of orthogonal matrices
            D (Tensor): (bs,d) batch of eigenvalues of the covariance matrix
            dDdt (Tensor): (bs,d) batch of time derivatives of eigenvalues of the covariance matrix

        Returns:
            Tensor: (..., bs, d) batch of drift function evaluations
        """
        # computing time derivative of K
        dKdt = self._dKdt_fast(U, dUdt, D, dDdt)
        p_noise = self.diffusion_prior.process_noise
        if dKdt.shape[0] != p_noise.shape[0] and p_noise.dim() == 3:
            p_noise = p_noise.unsqueeze(1)
        A = solve_lyapunov_spectral(D, U, -dKdt + p_noise)
        # computing mean and time derivitve of mean
        m, dmdt = self.m(t, return_grad=True)
        # squeeze for compat with integration schemes
        return dmdt - (A @ (z - m).unsqueeze(-1)).squeeze()

    def unweighted_residual_loss(self, drift: Tensor, f: Tensor) -> Tensor:
        """
        Computes the (unweighted) residual loss between the approximating and prior SDEs.

        Args:
            drift (Tensor): (n_samples, bs, d) drift function evaluations
            f (Tensor): (n_samples, bs, d) / (bs, d) prior SDE evaluations

        Returns:
            Tensor: residual loss
        """
        r = drift - f
        return (r.mul(self.diffusion_prior.solve_linear(r))).sum(-1)

    def generate_samples(
        self, t: Tensor, num_samples: int, return_intermediates: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]:
        """
        Generates samples from the approximating SDE marginal distribution at time t and optionally
        returns some intermediate quantities.

        Args:
            t (Tensor): (bs, ) time stamps at which to generate samples
            num_samples (int): number of independent samples at each time stamp
            return_intermediates (bool, optional): indicates whether to return intermediate quanties. Defaults to False.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]]: samples of latent states or samples of latent states and intermediate quantities
        """
        # get covariance at each time stamp
        if return_intermediates:
            U, dUdt = self.orthogonal(t, return_grad=True)
            D, dDdt = self.eigenvals(t, return_grad=True)
        else:
            U = self.orthogonal(t, return_grad=False)
            D = self.eigenvals(t, return_grad=False)
        # (len(t), d, d)
        L = U.mul(D.sqrt().unsqueeze(-2))
        # should I generate samples for at each time stamp, or can they be shared between stamps
        # for now we will generate independent samples at each time stamp
        v = torch.randn(num_samples, len(t), self.d, 1)
        # get mean at each time stamp
        mu = self.mean(t)
        zs = mu + (L @ v).squeeze(-1)
        if return_intermediates:
            return zs, U, dUdt, D, dDdt
        else:
            return zs

    def forward(
        self, t: Tensor, f: Callable[[Tensor, Tensor], Tensor], num_samples: int
    ) -> Tensor:
        """
        Computes the (unweighted) residual loss between the approximating and prior SDEs.

        Args:
            t (Tensor): (bs, ) time stamps at which to generate samples
            f (Callable[[Tensor,Tensor],Tensor]): (t, z) -> (num_samples, bs, d) or (bs, d) prior SDE evaluations
            num_samples (int): number of reparameterized samples at each time stamp

        Returns:
            Tensor: (bs,) approximate residual loss
        """
        # compute residual loss making good use of intermediate quantities
        zs, U, dUdt, D, dDdt = self.generate_samples(
            t, num_samples, return_intermediates=True
        )
        drift = self._drift_fast(t, zs, U, dUdt, D, dDdt)
        f_samples = f(t, zs)
        return self.unweighted_residual_loss(drift, f_samples).mean(0)
