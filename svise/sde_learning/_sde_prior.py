import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn import functional
import numpy as np
import math
from typing import Union, Tuple, Callable, Iterator, List, Optional
from abc import ABC, abstractmethod
from ..utils import *
from ..variationalsparsebayes.sparse_glm import (
    SparseFeaturesLibrary,
    inverse_softplus,
    softplus,
)
from ..variationalsparsebayes import SVIHalfCauchyPrior
from ..variationalsparsebayes.sparse_glm import SparsePolynomialFeatures

__all__ = [
    "SDEPrior",
    "AutonomousDriftFCNN",
    "SparseMultioutputGLM",
    "SparseIntegratorGLM",
    "SparseNeighbourGLM",
    "ExactMotionModel",
    "DriftFCNN",
]


class SDEPrior(ABC, nn.Module):
    """
    Abstract base class for SDE priors. (i.e. any subclass that inherits
    from this class and appropriately implements all methods is compatible
    with the SDELearner class).
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        """
        Compute derivatives under the variational distribution, q(theta) for a batch of latent states.
        Often this function will call self.resample_weights() followed by self.forward(t,z)

        Args:
            t (Tensor): (bs, ) time stamps at which to compute the drift
            z (Tensor): (n_reparam_samples, bs, d) batch of latent states

        Returns:
            Tensor: (n_reparam_samples, bs, d) batch of drift function evaluations
        """
        pass

    @abstractmethod
    def kl_divergence(self) -> Tensor:
        """
        Compute the KL divergence between the variational distribution, q(theta), and the prior distribution, p(theta).
        Zero is returned if doing maximum likelihood estimation.

        Returns:
            Tensor: kl divergence
        """
        pass

    @abstractmethod
    def resample_weights(self) -> None:
        """
        Resamples the weights from the variational distribution if applicable.
        """
        pass

    @abstractmethod
    def drift(self, t: Tensor, z: Tensor, integration_mode=False) -> Tensor:
        """
        Computes the derivatives for a given setting of the variational distribution weights
        """
        pass


class ExactMotionModel(SDEPrior):
    """
    A prior over the drift function for used in the case the
    exact form of the drift function is known (i.e. standard
    state estimation).

    Args:
        f (Callable[[Tensor, Tensor], Tensor]): exact drift function
    """

    def __init__(self, f: Callable[[Tensor, Tensor], Tensor]) -> None:
        super().__init__()
        self.f = f
        self.register_buffer("zero", torch.zeros(1))

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        """
        Compute derivatives under the variational distribution, q(theta) for a batch of latent states.

        Args:
            t (Tensor): (bs, ) time stamps at which to compute the drift
            z (Tensor): (n_reparam_samples, bs, d) batch of latent states

        Returns:
            Tensor: (n_reparam_samples, bs, d) batch of drift function evaluations
        """
        return self.f(t, z)

    def kl_divergence(self) -> Tensor:
        return self.zero

    def reparam_sample(self, n_reparam_samples: int) -> None:
        pass

    def drift(self, t: Tensor, z: Tensor, integration_mode=False) -> Tensor:
        return self.forward(t, z)

    def resample_weights(self) -> None:
        pass  # no weights to resample


class DriftFCNN(SDEPrior):
    """
    Class for inferring a fully connected neural network for the drift 
    function 

    Args:
        dim (int): dimension of the latent state
        layer_description (List[int]): list of hidden layer sizes
        nonlinearity (nn.Module): nonlinearity to use between layers
    """
    zero: Tensor

    def __init__(
        self, dim: int, layer_description: List[int], nonlinearity: nn.Module
    ) -> None:
        super().__init__()
        layers = [(nn.Linear(dim + 2, layer_description[0]), nonlinearity)]
        layers += [
            (nn.Linear(layer_description[i], layer_description[i + 1]), nonlinearity)
            for i in range(len(layer_description) - 1)
        ]
        layers = [mod for layer in layers for mod in layer]
        layers.append(nn.Linear(layer_description[-1], dim))
        self.layers = nn.Sequential(*layers)
        self.register_buffer("zero", torch.zeros(1))

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        t = torch.ones(x.shape[:-1], device=self.zero.device).mul(t).unsqueeze(-1)
        # suggested here: https://github.com/google-research/torchsde/blob/master/examples/latent_sde.py
        # Positional encoding in transformers for time-inhomogeneous posterior.
        return self.layers(torch.cat((torch.cos(t), torch.sin(t), x),dim=-1))

    def kl_divergence(self) -> Tensor:
        return self.zero

    def drift(self, t: Tensor, z: Tensor, integration_mode=False) -> Tensor:
        return self.forward(t, z)

    def resample_weights(self) -> None:
        pass


class AutonomousDriftFCNN(SDEPrior):
    """DO NOT USE DEPRECATED"""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        nonlinearity: Callable = nn.Softplus(),
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        # todo: is batch norm reasonable with variational inference?
        self.fcnn = FCNN(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=input_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            batch_norm=batch_norm,
        )
        self.register_buffer("zero", torch.zeros(1))

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        """
        Compute derivatives under the variational distribution, q(theta) for a batch of latent states.

        Args:
            t (Tensor): (bs, ) time stamps at which to compute the drift
            z (Tensor): (n_reparam_samples, bs, d) batch of latent states

        Returns:
            Tensor: (n_reparam_samples, bs, d) batch of drift function evaluations
        """
        return self.fcnn(z)

    def kl_divergence(self) -> Tensor:
        return self.zero

    def reparam_sample(self, n_reparam_samples: int) -> None:
        pass

    def drift(self, t: Tensor, z: Tensor, integration_mode=False) -> Tensor:
        return self.forward(t, z)

    def resample_weights(self) -> None:
        pass  # no weights to resample


class SparseMultioutputGLM(SDEPrior):
    """
    Sparse multioutput generalized linear model for the drift function.

    Args:
        d (int): Dimension of output.
        SparseFeatures (SparseFeaturesLibrary): Sparse features library.
        n_reparam_samples (int): Number of reparameterized samples.
        tau (float, optional): Prior on global scaling parameter. Defaults to 1e-5.
        train_x (Tensor, optional): Training states (for initialization)
        train_y (Tensor, optional): Training state derivatives (for intialization)
        resample_on_init (bool, optional): boolean indicating whether to resample weights after initialization
        transform (ScaleTransform, optional): NOT TESTED, DO NOT USE
    """

    def __init__(
        self,
        d: int,
        SparseFeatures: SparseFeaturesLibrary,
        n_reparam_samples: int,
        tau: float = 1e-5,
        train_x: Tensor = None,
        train_y: Tensor = None,
        resample_on_init: bool = True,
        transform: ScaleTransform = None,
    ):
        super().__init__()
        self.d = d
        if transform is None:
            self.transform = IdentityScaleTransform(d)
        else:
            self.transform = transform
        self.features = SparseFeatures
        num_weights = d * self.features.num_features
        self.num_features = self.features.num_features
        self.n_reparam_samples = n_reparam_samples
        self.register_buffer("zero", torch.zeros(num_weights))
        self.momentum = 0.1
        self.register_buffer("eps", torch.tensor(1e-5))
        self.update_running_stats = True
        if train_x is not None:
            assert train_y is not None, "train_x and train_y must both be provided"
            self.update_running_stats = False
            train_x = self.transform.inverse(train_x)
            self.register_buffer("running_var", self.features(train_x).var(0).detach())
            self.running_var[self.running_var == 0] = 1.0
            # initialize with least squares
            phi_x = self._tfm(self.features(train_x))
            W = solve_least_squares(phi_x, train_y, gamma=1e-1).t().flatten().detach()
            # initialize with uwbayes
            # W = torch.from_numpy(solve_uwbayes(train_t, train_x, poly_degree=5))
            # W = self.transform(W)
            # W = (W.t() * self.running_var.sqrt()).flatten().detach()
            self.prior = SVIHalfCauchyPrior(d=num_weights, tau=tau, w_init=W)
            self.update_running_stats = True
        elif train_y is not None:
            raise ValueError("train_x and train_y must both be provided.")
        else:
            self.register_buffer("running_var", torch.ones(self.num_features))
            self.prior = SVIHalfCauchyPrior(d=num_weights, tau=tau)
        if resample_on_init:
            self.resample_weights()

    @property
    def W(self) -> Tensor:
        return self.__W

    @W.setter
    def W(self, value: Tensor) -> None:
        tmp = self.zero.repeat(value.shape[0], 1)
        tmp[:, self.prior.sparse_index] = value
        self.__W = tmp.reshape(-1, self.d, self.num_features)

    def _tfm(self, x: Tensor) -> Tensor:
        if self.training and self.update_running_stats:
            with torch.no_grad():
                bs = x.shape[:-1]
                dims = [i for i in range(len(bs))]
                counter = np.prod(bs)
                x_sum = x.sum(dims)
                x_sum_squares = (x**2).sum(dims)
                mean = x_sum / counter
                var = x_sum_squares / counter - mean**2
                var[var == 0] = 1.0
                m = self.momentum
                self.running_var = (1 - m) * self.running_var + m * var
                self.running_var[self.running_var == 0] = 1.0
        return x / self.running_var.add(self.eps).sqrt()

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        self.resample_weights()
        return self.drift(t, z)

    def resample_weights(self) -> None:
        w_sample = self.prior.get_reparam_weights(self.n_reparam_samples)
        self.W = w_sample

    def drift(self, t: Tensor, z: Tensor, integration_mode=False) -> Tensor:
        z = self.transform.inverse(z)
        phi_z = self._tfm(self.features(z))
        if integration_mode:
            # todo: we need to check this theoretically
            return torch.einsum("ijk,ik->ij", self.W, phi_z)
        if z.dim() == 3:
            dz = (
                (self.W @ phi_z.transpose(0, 1).unsqueeze(-1)).transpose(0, 1).squeeze()
            )
        else:
            dz = (self.W @ phi_z.t()).transpose(-2, -1)
        return dz

    def kl_divergence(self) -> Tensor:
        return self.prior.kl_divergence()

    def update_sparse_index(self) -> None:
        return self.prior.update_sparse_index()

    def reset_sparse_index(self) -> None:
        self.prior.reset_sparse_index()

    @property
    def feature_names(self) -> List[str]:
        return self.get_feature_names(num=2)

    def get_feature_names(self, num: int = 5) -> List[str]:
        fnames = np.array(self.features.feature_names)
        all_names = np.stack([fnames.copy() for _ in range(self.d)])
        sparse_index = self.prior.sparse_index.clone().numpy().reshape(self.d, -1)
        flip_update_running_stats = self.update_running_stats
        if self.update_running_stats:
            self.update_running_stats = False
        weights = self.transform.inverse(self._tfm(self.W.mean(0)).t()).t()
        if flip_update_running_stats:
            self.update_running_stats = True
        return_names = []
        for j in range(all_names.shape[0]):
            name_list = [
                f"{w:.{num}f}{n}"
                for w, n in zip(
                    weights[j, sparse_index[j]], all_names[j, sparse_index[j]]
                )
            ]
            return_names.append(" + ".join(name_list))
        return return_names


class SparseIntegratorGLM(SparseMultioutputGLM):
    """
    Assumes the governing equations can be written in the form: d^2x/dt^2 = f(x, dx/dt),
    where f is a sparse linear combination of functions from the features library.

    Args:
        d (int): Dimension of output.
        SparseFeatures (SparseFeaturesLibrary): Sparse features library.
        n_reparam_samples (int): Number of reparameterized samples.
        integrator_indices (List[int]): which states correspond to unknown dynamics
        tau (float, optional): Prior on global scaling parameter. Defaults to 1e-5.
        train_x (Tensor, optional): Training states (for initialization)
        train_y (Tensor, optional): Training state derivatives (for intialization)
    """

    def __init__(
        self,
        d: int,
        SparseFeatures: SparseFeaturesLibrary,
        n_reparam_samples: int,
        integrator_indices: List[int] = None,
        tau: float = 1e-5,
        train_x: Tensor = None,
        train_y: Tensor = None,
    ):
        if integrator_indices is None:
            # assumes default polynomial dictionary with bias
            integrator_indices = [d // 2 + 1 + j for j in range(d // 2)]
        I = torch.eye(SparseFeatures.num_features)
        assert d % 2 == 0, "d must be even."
        if train_y is not None:
            train_y = train_y[..., d // 2 :]

        super().__init__(
            d // 2,
            SparseFeatures,
            n_reparam_samples,
            tau,
            train_x,
            train_y,
            resample_on_init=False,
        )
        self.d = d
        integrator_matrix = torch.stack(
            [I[integrator_indices[j]] for j in range(d // 2)], dim=0
        ).unsqueeze(0)
        self.register_buffer("integrator_matrix", integrator_matrix)
        self.resample_weights()

    @property
    def W(self) -> Tensor:
        W_tmp = self.__W.clone()
        # accounting for the scaling in V
        W_tmp[:, : self.d // 2] = (
            W_tmp[:, : self.d // 2] * self.running_var.add(self.eps).sqrt()
        )
        return W_tmp

    @W.setter
    def W(self, value: Tensor) -> None:
        nsamples = value.shape[0]
        tmp = self.zero.repeat(nsamples, 1)
        # todo: what to do with sparse index??
        tmp[:, self.prior.sparse_index] = value
        tmp = torch.cat(
            [
                self.integrator_matrix.repeat(nsamples, 1, 1),
                tmp.reshape(-1, self.d // 2, self.num_features),
            ],
            dim=1,
        )
        self.__W = tmp

    def get_feature_names(self, num: int = 5, scale: Tensor = None) -> List[str]:
        # todo: fix this
        if scale is None:
            scale = torch.ones(self.d // 2)
        fnames = np.array(self.features.feature_names)
        all_names = np.stack([fnames.copy() for _ in range(self.d // 2)])
        sparse_index = self.prior.sparse_index.clone().numpy().reshape(self.d // 2, -1)
        weights = self._tfm((self.W.mean(0)[self.d // 2 :].t().mul(scale)).t())
        feature_names = []
        for j in range(all_names.shape[0]):
            name_list = [
                f"{w:.{num}f}{n}"
                for w, n in zip(
                    weights[j, sparse_index[j]], all_names[j, sparse_index[j]]
                )
            ]
            feature_names.append(" + ".join(name_list))
        speed_names = [
            self.features.input_labels[j] for j in range(self.d // 2, self.d)
        ]
        integrator_labels = [f"{1.0:.{num}f}{n}" for n in reversed(speed_names)]
        [feature_names.insert(0, ilabel) for ilabel in integrator_labels]
        return feature_names


class SparseNeighbourGLM(SDEPrior):
    """
    Sparse multioutput generalized linear model where the drift is assumed
    to be a function of its neighbours.


    Args:
        d (int): Dimension of output.
        SparseFeatures (SparseFeaturesLibrary): Sparse features library.
        n_reparam_samples (int): Number of reparameterized samples.
        tau (float, optional): Prior on global scaling parameter. Defaults to 1e-5.
        train_x (Tensor, optional): Training states (for initialization)
        train_y (Tensor, optional): Training state derivatives (for intialization)
        resample_on_init (bool, optional): boolean indicating whether to resample weights after initialization
        transform (ScaleTransform, optional): NOT TESTED, DO NOT USE
    """

    def __init__(
        self,
        d: int,
        SparseFeatures: SparseFeaturesLibrary,
        n_reparam_samples: int,
        tau: float = 1e-5,
        train_x: Tensor = None,
        train_y: Tensor = None,
        resample_on_init: bool = True,
        transform: ScaleTransform = None,
    ):
        super().__init__()
        self.d = d
        if transform is None:
            self.transform = IdentityScaleTransform(d)
        else:
            self.transform = transform
        self.features = SparseFeatures
        num_weights = self.features.num_features
        self.num_features = self.features.num_features
        self.n_reparam_samples = n_reparam_samples
        self.register_buffer("zero", torch.zeros(num_weights))
        self.momentum = 0.1
        self.register_buffer("eps", torch.tensor(1e-5))
        self.update_running_stats = True
        if train_x is not None:
            assert train_y is not None, "train_x and train_y must both be provided"
            self.update_running_stats = False
            train_x = self.transform.inverse(train_x)
            self.register_buffer(
                "running_var", self.features(train_x).var((0, 1)).detach()
            )
            self.running_var[self.running_var == 0] = 1.0
            # initialize with least squares
            # todo: double check that this reshaping is preserving the order of the data
            phi_x = self._tfm(self.features(train_x)).reshape(-1, self.num_features)
            train_y = train_y.reshape(-1, 1)
            W = solve_least_squares(phi_x, train_y, gamma=1e-1).t().flatten().detach()
            self.prior = SVIHalfCauchyPrior(d=num_weights, tau=tau, w_init=W)
            self.update_running_stats = True
        elif train_y is not None:
            raise ValueError("train_x and train_y must both be provided.")
        else:
            self.register_buffer("running_var", torch.ones(self.num_features))
            self.prior = SVIHalfCauchyPrior(d=num_weights, tau=tau)
        if resample_on_init:
            self.resample_weights()

    @property
    def W(self) -> Tensor:
        return self.__W

    @W.setter
    def W(self, value: Tensor) -> None:
        tmp = self.zero.repeat(value.shape[0], 1)
        tmp[:, self.prior.sparse_index] = value
        self.__W = tmp.reshape(-1, self.num_features)

    def _tfm(self, x: Tensor) -> Tensor:
        if self.training and self.update_running_stats:
            with torch.no_grad():
                bs = x.shape[:-1]
                dims = [i for i in range(len(bs))]
                counter = np.prod(bs)
                x_sum = x.sum(dims)
                x_sum_squares = (x**2).sum(dims)
                mean = x_sum / counter
                var = x_sum_squares / counter - mean**2
                var[var == 0] = 1.0
                m = self.momentum
                self.running_var = (1 - m) * self.running_var + m * var
                self.running_var[self.running_var == 0] = 1.0
        return x / self.running_var.add(self.eps).sqrt()

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        self.resample_weights()
        return self.drift(t, z)

    def resample_weights(self) -> None:
        w_sample = self.prior.get_reparam_weights(self.n_reparam_samples)
        self.W = w_sample

    def drift(self, t: Tensor, z: Tensor, integration_mode=False) -> Tensor:
        z = self.transform.inverse(z)
        phi_z = self._tfm(self.features(z))
        if integration_mode:
            raise NotImplementedError
            # todo: we need to check this theoretically
            # return torch.einsum("ijk,ik->ij", self.W, phi_z)
        if z.dim() == 3:
            dz = torch.einsum("ijkl,il->ijk", phi_z, self.W)
            # dz = (self.W.unsqueeze(1) @ phi_z.transpose(0,2).unsqueeze(-1)).squeeze().transpose(0,2)
        else:
            # raise NotImplementedError
            # dz = (self.W @ phi_z.t()).transpose(-2, -1)
            dz = torch.einsum("il,jkl->ijk", self.W, phi_z)
        return dz

    def kl_divergence(self) -> Tensor:
        return self.prior.kl_divergence()

    def update_sparse_index(self) -> None:
        return self.prior.update_sparse_index()

    def reset_sparse_index(self) -> None:
        self.prior.reset_sparse_index()
        self.features.update_basis(self.prior.sparse_index)

    @property
    def feature_names(self) -> List[str]:
        return self.get_feature_names(num=2)

    def get_feature_names(self, num: int = 5) -> List[str]:
        fnames = np.array(self.features.feature_names)
        # all_names = np.stack([fnames.copy() for _ in range(self.d)])
        sparse_index = self.prior.sparse_index.clone().numpy()
        flip_update_running_stats = self.update_running_stats
        if self.update_running_stats:
            self.update_running_stats = False
        weights = self.transform.inverse(self._tfm(self.W.mean(0)))
        if flip_update_running_stats:
            self.update_running_stats = True
        sparse_weights = weights[sparse_index]
        sparse_names = fnames[sparse_index]
        name_list = "+".join(
            [f"{w:.{num}f}{n}" for w, n in zip(sparse_weights, sparse_names)]
        )
        return_names = [name_list]
        return return_names
