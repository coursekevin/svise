import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch import Tensor
from torch.nn import functional
from abc import ABC, abstractmethod
from ..utils import *
from ..variationalsparsebayes import SVIHalfCauchyPrior

__all__ = [
    "DiffusionPrior",
    "DiagonalDiffusionPrior",
    "ConstantDiffusionPrior",
    "MLScaleDiffusionPrior",
    "SparseDiagonalDiffusionPrior",
    "ConstantDiagonalDiffusionPrior",
]


class DiffusionPrior(ABC, nn.Module):
    """
    Abstract base class for diffusion prior. This class assumes that diffusion is constant.

    Args:
        d (int): number of states
        Q (Tensor): Diffusion matrix init
    """

    def __init__(self, d: int, Q: Tensor) -> None:
        super().__init__()
        self.d = d
        assert Q.shape == (d, d), "Q is expected to be a constant square matrix."
        self.register_buffer("Q", Q)
        self.register_buffer("Q_chol", torch.linalg.cholesky(Q))

    @abstractmethod
    def resample_weights(self) -> None:
        """Uses reparam. trick to update any parameters if applicable."""
        pass

    @property
    @abstractmethod
    def process_noise(self) -> Tensor:
        """Returns the full process nosie matrix (L Q L^T )

        Returns:
            Tensor: full process noise matrix
        """
        pass

    @property
    @abstractmethod
    def kl_divergence(self) -> Tensor:
        """Returns the kl-divergence between the approx. posterior
        and the prior

        Returns:
            Tensor: kl-div between approx. posterior and the prior
        """
        pass

    @abstractmethod
    def solve_linear(self, r: Tensor) -> Tensor:
        """Solves the linear system L Q L^T x = r
            r (Tensor): residual

        Returns:
            Tensor: x = (LQL^T)^{-1} r
        """
        pass


class DiagonalDiffusionPrior(DiffusionPrior):
    """
    Base class for the case that the diffusion matrix is diagonal.

    Args:
        d (int): number of states
        Q (Tensor): Diffusion matrix init
    """

    def __init__(self, d: int, Q: Tensor) -> None:
        super().__init__(d, Q)

    @property
    @abstractmethod
    def process_noise_diag(self) -> Tensor:
        """Returns the diagonal elements of the process noise term

        Returns:
            Tensor: returns diagonal element of the process noise term
        """
        pass


class ConstantDiagonalDiffusionPrior(DiagonalDiffusionPrior):
    """Class for the case the diffusion matrix is known
    and constant (i.e. won't be tuned during training):

    Args:
        d (int): number of states
        Q_diag (Tensor): diagonal component of diffusion matrix
    """

    def __init__(self, d: int, Q_diag: Tensor) -> None:
        assert Q_diag.shape == (d,), "Q_diag is expected to be a vector."
        super().__init__(d, Q_diag.diag())
        self.register_buffer("Q_diag", Q_diag)
        self.register_buffer("pnoise", Q_diag.diag())
        self.register_buffer("zero", torch.zeros(1))

    def resample_weights(self) -> None:
        return None

    @property
    def process_noise(self) -> Tensor:
        return self.pnoise

    @property
    def process_noise_diag(self) -> Tensor:
        return self.Q_diag

    @property
    def kl_divergence(self) -> Tensor:
        return self.zero

    def solve_linear(self, r: Tensor) -> Tensor:
        # p_noise_diag = self.Q_diag.unsqueeze(1)
        # return r.div(p_noise_diag)
        return r.div(self.Q_diag)


class ConstantDiffusionPrior(DiffusionPrior):
    """DO NOT USE, NOT PROPERLY TESTED"""

    def __init__(self, d: int, Q: Tensor, Sigma: Tensor) -> None:
        super().__init__(Q.shape[0], Q)
        U, S, Vh = torch.linalg.svd(Sigma @ self.Q_chol, full_matrices=False)
        self.register_buffer("U", U)
        self.register_buffer("S", S)
        self.register_buffer("V", Vh.T)
        self.register_buffer("pnoise", (U * S.pow(2)) @ U.T)
        self.register_buffer("sigma_matrix", Sigma)
        self.register_buffer("zero", torch.zeros(1))

    def resample_weights(self) -> None:
        return None
        # raise NotImplementedError("ConstantDiffusionPrior cannot be resampled.")

    @property
    def process_noise(self) -> Tensor:
        return self.pnoise

    @property
    def sigma(self) -> Tensor:
        return self.sigma_matrix

    @property
    def kl_divergence(self) -> Tensor:
        return self.zero

    def solve_linear(self, r: Tensor) -> Tensor:
        U, S = self.U, self.S
        return r @ (U * S.pow(-2)) @ U.T


# TODO: test this class
class MLScaleDiffusionPrior(DiffusionPrior):
    def __init__(self, d: int) -> None:
        Q = torch.eye(d)
        super().__init__(d, Q)
        self.raw_scale = Parameter(torch.ones(d) * 0.5413)
        self.register_buffer("zero", torch.zeros(1))

    def resample_weights(self) -> None:
        pass

    @property
    def scale(self) -> Tensor:
        return functional.softplus(self.raw_scale)

    @property
    def process_noise(self) -> Tensor:
        return self.scale.diag()

    @property
    def kl_divergence(self) -> Tensor:
        return self.zero

    def solve_linear(self, r: Tensor) -> Tensor:
        return r.div(self.scale)


class SparseDiagonalDiffusionPrior(DiagonalDiffusionPrior):
    """Prior over the diffusion matrix is a sparse diagonal matrix.

    Args:
        d (int): number of states
        Q_diag (Tensor): the starting value of the dispersion matrix
        n_reparam_samples (int): number of reparametrization samples to use
        tau (float): global scaling parameter
    """

    def __init__(
        self, d: int, Q_diag: Tensor, n_reparam_samples: int, tau: float
    ) -> None:
        assert Q_diag.shape == (d,), "Q_diag is expected to be a vector."
        super().__init__(d, Q_diag.diag())
        self.n_reparam_samples = n_reparam_samples
        self.register_buffer("Q_diag", Q_diag)
        self.prior = SVIHalfCauchyPrior(d=d, tau=tau, w_init=torch.ones(d))
        self.resample_weights()

    @property
    def Sigma_diag(self) -> Tensor:
        return self._Sigma_diag

    @Sigma_diag.setter
    def Sigma_diag(self, value: Tensor) -> None:
        self._Sigma_diag = value

    @property
    def process_noise(self) -> Tensor:
        return self.Sigma_diag.pow(2).mul(self.Q_diag).diag_embed()

    @property
    def process_noise_diag(self) -> Tensor:
        return self.Sigma_diag.pow(2).mul(self.Q_diag)

    @property
    def kl_divergence(self) -> Tensor:
        return self.prior.kl_divergence()

    def resample_weights(self) -> None:
        # note: i don't think pow(2) is needed because we always square sigma first
        Sigma_diag_sample = self.prior.get_reparam_weights(self.n_reparam_samples)
        self.Sigma_diag = Sigma_diag_sample

    def solve_linear(self, r: Tensor) -> Tensor:
        p_noise_diag = self.Sigma_diag.pow(2).mul(self.Q_diag).unsqueeze(1)
        return r.div(p_noise_diag)
