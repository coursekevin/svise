from __future__ import annotations

import logging
import math
from typing import Any, Mapping, Tuple, Optional

import torch
from torch import Tensor
from torch import nn
from sklearn.utils.extmath import randomized_svd

__all__ = [
    "PCA",
    "pca_log_mll",
]


def pca_log_mll(rank, spectrum, num_data):
    """
    follows the formula (30) in https://proceedings.neurips.cc/paper/2000/file/7503cfacd12053d309b6bed5c89de212-Paper.pdf.
    also used by sklearn to automatically infer rank
    """
    dim = len(spectrum)
    # m in the paper
    manifold_dim = dim * rank - rank * (rank + 1) / 2
    # prior over u
    evidence = -rank * math.log(2)
    range_vec = (dim - (torch.arange(rank) + 1) + 1) / 2
    evidence += torch.lgamma(range_vec).sum()
    evidence += -range_vec.sum() * math.log(math.pi)
    # lambda prod
    evidence += torch.log(spectrum[:rank]).sum().mul(-num_data / 2)
    # v_hat
    evidence += torch.log(spectrum[rank:].sum() / (dim - rank)).mul(
        -num_data * (dim - rank) / 2
    )
    evidence += torch.log(2 * torch.tensor(torch.pi)).mul((manifold_dim + rank) / 2)
    # |A_z|^{-1/2}
    spectrum_hat = spectrum.clone()
    spectrum_hat[rank:] = (spectrum[rank:]).sum() / (dim - rank)
    for i in range(rank):
        evidence += -0.5 * torch.log(spectrum[i] - spectrum[i + 1 :]).sum()
        evidence += (
            -0.5 * torch.log(1 / spectrum_hat[i + 1 :] - 1 / spectrum_hat[i]).sum()
        )
        evidence += -0.5 * (dim - (i + 1)) * math.log(num_data)
    evidence += torch.log(torch.tensor(num_data)).mul(-rank / 2)
    return evidence


class PCA(nn.Module):
    """Class for performing PCA with automatic selection
    of the rank of the using the method from
    https://proceedings.neurips.cc/paper/2000/file/7503cfacd12053d309b6bed5c89de212-Paper.pdf.
    """

    def __init__(
        self,
        evecs: Optional[Tensor] = None,
        rank: Optional[Tensor] = None,
        mean: Optional[Tensor] = None,
        scale: Optional[Tensor] = None,
    ) -> None:
        super().__init__()
        if evecs is None:
            # if no data is provided init some empty buffers
            # this should only be called if we are loading from memory
            self.init_buffers()
        else:
            self.register_buffer("mean", mean)
            self.register_buffer("n_components", rank)
            self.register_buffer("evecs", evecs)
            self.register_buffer("scale", scale)

    def init_buffers(self, n_dim: int = 0, n_components: int = 0):
        """Init empty buffers (useful for loading from memory when
        size of buffers might be unknown )
        """
        self.register_buffer("mean", torch.empty(n_dim))
        self.register_buffer("evecs", torch.empty(n_dim, n_components))
        self.register_buffer("n_components", torch.tensor(n_components))
        self.register_buffer("scale", torch.ones(n_components))

    @classmethod
    def create(
        cls,
        y: Tensor,
        percent_cutoff: float = 0.95,
        max_evecs: int = 30,
        rescale: bool = True,
    ) -> Tuple[PCA, Tensor]:
        """Initialize a pca decomposition and return the transformed
        code vectors

        Args:
            y (Tensor): (N,D) input data
            percent_cutoff (float): what percentage to use as a cut off when
                getting a rough estimate for the rank of the covariance matrix
            max_evecs (int): maximum number of eigenvectors to compute
            rescale (bool): whether to rescale the code vectors so that variance of the max is 1

        Returns:
            pca_model, z
        """

        z, rank, evecs, mean = cls._fit_init(
            y=y, percent_cutoff=percent_cutoff, max_evecs=max_evecs
        )
        scale = torch.ones(z.shape[1])
        if rescale:
            scale = z.std(0).max() * scale
            z = z / scale
        lin_model = cls(evecs=evecs, rank=rank, mean=mean, scale=scale)
        return lin_model, z

    @classmethod
    def _fit_init(cls, y: Tensor, percent_cutoff: float, max_evecs: int):
        """Comutes pca transformation selecting rank using
        pca_log_mll ranking. Used sklearn randomized_svd to compute
        eigenvectors and eigenvalues
            y (Tensor): Input data
            percent_cutoff (float): what percentage of variance
            to use for initial rank cut_off
            max_evecs (int): maximum number of eigenvectors to compute

        Returns:
            z, rank, evecs, mean (code vecs, selected rank, evecs, and mean)
        """
        mean = y.mean(0)

        with torch.no_grad():
            max_evecs = min(max_evecs, *y.shape)
            u, s, vh = randomized_svd(
                (y - mean).numpy(), n_components=max_evecs, random_state=None
            )
            u, s, vh = torch.as_tensor(u), torch.as_tensor(s), torch.as_tensor(vh)

        variance = s.pow(2) / (y.shape[0] - 1)
        # total_variance = variance.sum()
        # compute this way in the case that we can't compute all singular values
        total_variance = (y - mean).pow(2).sum() / (y.shape[0] - 1)
        # percentage of variance explained by principle component
        percent_var = torch.cumsum(variance / total_variance, dim=0)
        log_mll = []
        for j, var in enumerate(percent_var):
            rank_var = j + 1
            lml = pca_log_mll(j + 1, variance, y.shape[0])
            if lml == torch.inf:
                break
            else:
                log_mll.append(lml)
            if var > percent_cutoff:
                logging.info(
                    "Auto dim reduction converged due to percent_cutoff %s",
                    percent_cutoff,
                )
                break
            if rank_var == max_evecs:
                logging.info(
                    "Auto dim reduction converged by reaching max_evecs %s",
                    max_evecs,
                )
                break
        # best rank based on log_mll
        rank = torch.tensor(log_mll).max(0).indices + 1
        if rank < rank_var:
            logging.info("Selecting rank based on log mll.")
        logging.info(
            "PCA converged to rank %s, capturing %s of variance",
            rank,
            percent_var[rank - 1],
        )
        z = u[:, :rank].mul(s[:rank])
        evecs = vh.T[:, :rank]
        return z, rank, evecs, mean

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        """Standard call to load_state_dict where buffers are first
        made to be the correct size
        """
        n_dim, n_components = state_dict["evecs"].shape
        self.init_buffers(n_dim=n_dim, n_components=n_components)
        return super().load_state_dict(state_dict, strict)

    def encode(self, y: Tensor) -> Tensor:
        """Encode a set of inputs (bs, d) into the lower dimensional space

        Args:
            y (Tensor): inputs

        Returns:
            reduced dimension inputs
        """
        return (y - self.mean) @ self.evecs / self.scale

    def decode(self, z: Tensor) -> Tensor:
        """Decode code variables (bs, n_components) -> (bs, d )

        Args:
            z (Tensor ): code variables

        Returns:
            approximation to y = decode(encode(y))
        """
        return self.forward(z)

    def forward(self, z: Tensor) -> Tensor:
        """Alias for decode"""
        return (z * self.scale) @ self.evecs.T + self.mean

    def transform_stdev(self, stdev: Tensor):
        """Transform the standard deviation of the input data into a variance on hte reduced order space

        Args:
            stdev (Tensor): (D) standard deviation of the input data

        Retruns:
            (Tensor): (n_components, n_components) variance of the reduced order space
        """
        # E[(z - E[z])(z - E[z])^T] = U^T E[eps eps^T] U
        assert stdev.dim() == 1, "only implemented for diagonal variance on y."
        U = self.evecs / self.scale
        return (U.T * stdev.pow(2)) @ U
