from abc import ABC, abstractmethod
from .svi_half_cauchy import SVIHalfCauchyPrior
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
import math
from typing import Tuple, Dict, Union, List, Iterable, Callable
from itertools import chain, combinations_with_replacement, tee
import numpy as np


def isotropic_gaussian_loglikelihood(
    x: Tensor, mu: Tensor, var: Tensor, N: int
) -> Tensor:
    logsigma = torch.log(var.sqrt())
    log2pi = torch.log(torch.tensor(2 * math.pi))
    k = x.shape[-1]
    diff = (x - mu).pow(2).sum(-1)
    return -N / (2 * var) * diff.mean() - N * k / 2 * log2pi - N * k * logsigma


def softplus(x: Tensor) -> Tensor:
    return torch.log(1 + torch.exp(x))


def inverse_softplus(x: Tensor) -> Tensor:
    return torch.log(torch.exp(x) - 1)


class SparseFeaturesLibrary(ABC, nn.Module):
    """
    Abstract base class for making sparse feature libraries
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def update_basis(self, sparse_index: Tensor) -> None:
        assert (
            self.num_features >= sparse_index.max()
        ), "More sparsity inducing indices than features."
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass

    @property
    @abstractmethod
    def feature_names(self) -> List[str]:
        pass

    @property
    @abstractmethod
    def num_features(self) -> int:
        pass


class SparsePrecomputedFeatures(SparseFeaturesLibrary):
    def __init__(self, num_features: int, input_labels: np.ndarray = None) -> None:
        super().__init__()
        self.num_features = num_features
        if input_labels is None:
            self.feature_names = np.array(
                ["x_{}".format(i) for i in range(num_features)]
            )
        else:
            assert (
                len(input_labels) == num_features
            ), "Number of labels should be the same as number of inputs."
            self.feature_names = input_labels
        self.sparse_index = torch.arange(self.num_features)

    @property
    def feature_names(self) -> List[str]:
        return self.__feature_names

    @feature_names.setter
    def feature_names(self, value):
        self.__feature_names = value

    @property
    def num_features(self) -> int:
        return self.__num_features

    @num_features.setter
    def num_features(self, value):
        self.__num_features = value

    def update_basis(self, sparse_index: Tensor) -> None:
        """
        Updates the spares_index 

        Args:
            sparse_index (Tensor): sparsity inducing indices 
        """
        super().update_basis(sparse_index)
        self.sparse_index = sparse_index

    def __str__(self) -> str:
        return " + ".join(self.feature_names[self.sparse_index])

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the sparse dictionary of basis functions output

        Args:
            x (Tensor): input (..., num_features)

        Returns:
            Tensor: sparse dictionary evaluated at input (..., num_features[sparse_index])
        """
        return x[..., self.sparse_index]


class Polynomial:
    """
    Class for computing polynomial feature of a given degree
    """

    def __init__(self, ind) -> None:
        self.ind = ind

    def __call__(self, x: Tensor) -> Tensor:
        return x[..., self.ind].prod(-1)


class SparsePolynomialFeatures(SparseFeaturesLibrary):
    """
    Polynomial features library for use with sparse models. Draws heavily from scikit learn implementation.
    """

    def __init__(
        self,
        dim: int,
        degree: int,
        include_bias: bool = True,
        input_labels: List[str] = None,
    ):
        super().__init__()
        self.dim = dim
        self.degree = degree
        start = int(not include_bias)
        iter_comb = chain.from_iterable(
            combinations_with_replacement(range(dim), i)
            for i in range(start, degree + 1)
        )
        # create two copies of iterable
        iter_comb_1, iter_comb_2 = tee(iter_comb)
        # array of functions
        self.full_polynomial = np.array([Polynomial(ind) for ind in iter_comb_1])
        self.num_features = len(self.full_polynomial)
        self.register_buffer("sparse_index", torch.arange(self.num_features))
        # getting names (taken directly from sklearn)
        if input_labels is None:
            self.input_labels = ["x_{}".format(i) for i in range(dim)]
        else:
            assert len(input_labels) == dim
            self.input_labels = input_labels
        self.feature_names = self._compute_feature_names(iter_comb_2)

    @property
    def feature_names(self) -> List[str]:
        return self.__feature_names

    @feature_names.setter
    def feature_names(self, value):
        self.__feature_names = value

    @property
    def num_features(self) -> int:
        return self.__num_features

    @num_features.setter
    def num_features(self, value):
        self.__num_features = value

    def update_basis(self, sparse_index: Tensor) -> None:
        """
        Updates the spares_index 

        Args:
            sparse_index (Tensor): sparsity inducing indices 
        """
        super().update_basis(sparse_index)
        self.sparse_index = sparse_index

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the sparse dictionary of basis functions output

        Args:
            x (Tensor): input (..., dim)

        Returns:
            Tensor: sparse dictionary evaluated at input (..., num_features[sparse_index])
        """
        return torch.stack(
            [f_i(x) for f_i in self.full_polynomial[self.sparse_index]], dim=-1
        )

    def __str__(self) -> str:
        return " + ".join(self.feature_names[self.sparse_index])

    def _compute_feature_names(self, iter_comb: Iterable) -> np.ndarray:
        """
        Compute a list of feature names. This function is taken directly from the sklearn implementation

        Args:
            iter_comb (Iterable): combination iterable for polynomials

        Returns:
            List[str]: list of strings containing feature names
        """
        powers = np.vstack([np.bincount(c, minlength=self.dim) for c in iter_comb])
        feature_names = []
        for row in powers:
            inds = np.where(row)[0]
            if len(inds):
                name = " ".join(
                    "%s^%d" % (self.input_labels[ind], exp)
                    if exp != 1
                    else self.input_labels[ind]
                    for ind, exp in zip(inds, row[inds])
                )
            else:
                name = "1"
            feature_names.append(name)
        return np.array(feature_names)


# TODO: unit test this class
class SparsePolynomialSinusoidTfm(SparsePolynomialFeatures):
    """
    Feature library that includes polynomials and sinusoids transform of input features.
    ex: ["x", "y"] -> ["1", "x", "y", "sin(x)", "cos(x)", "sin(y)", "cos(y)", "x^2", "y^2", ... ]

    """

    def __init__(
        self,
        dim: int,
        degree: int,
        include_bias: bool = True,
        input_labels: List[str] = None,
    ):
        # note, I don't think we need a freq + phase argument as we can always scale x?
        if input_labels is None:
            input_labels = ["x_{}".format(i) for i in range(dim)]
        else:
            input_labels = [el for el in input_labels]
        sin_labels = [f"sin({il})" for il in input_labels]
        cos_labels = [f"cos({il})" for il in input_labels]
        input_labels += sin_labels + cos_labels
        super().__init__(dim * 3, degree, include_bias, input_labels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the sparse dictionary of basis functions output

        Args:
            x (Tensor): input (..., dim)

        Returns:
            Tensor: sparse dictionary evaluated at input (..., num_features[sparse_index])
        """
        x_in = torch.cat([x, torch.sin(x), torch.cos(x)], dim=-1)
        return super().forward(x_in)


# TODO: unit test this class
class SparsePolynomialNeighbour1D(SparsePolynomialFeatures):
    """
    Polynomial feature library for use with sparse models. Assumes that each input is a polynomial function of it's neighbours.

    ex: ["x_0", "x_1", "x_2"] = ["x_0", "x_1", "x_2", "x_0 x_1", "x_1 x_2", "x_0^2", "x_1^2", "x_2^2"]
    """

    def __init__(
        self,
        dim: int,
        degree: int,
        include_bias: bool = True,
        input_labels: List[str] = None,
        num_neighbours: int = 2,
    ):
        assert dim >= 3
        self.num_neighbours = num_neighbours
        inds = [i - self.num_neighbours for i in range(2 * self.num_neighbours + 1)]
        self.inds = inds
        if input_labels is None:
            var_name = ["x"]
        else:
            assert len(input_labels) == 1
            var_name = input_labels[0]
        input_labels = [f"{var_name}_{{k{-i:+}}}" for i in inds]
        super().__init__(2 * num_neighbours + 1, degree, include_bias, input_labels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Computes the sparse dictionary of basis functions output

        Args:
            x (Tensor): input (..., dim)

        Returns:
            Tensor: sparse dictionary evaluated at input (..., num_features[sparse_index])
        """
        x_in = torch.stack([x.roll(i, dims=-1) for i in self.inds], dim=-1,)
        return super().forward(x_in)


class SparseGLMGaussianLikelihood(nn.Module):
    def __init__(
        self,
        d: int,
        SparseFeatures: SparseFeaturesLibrary,
        noise: float = 1e-3,
        learn_noise: bool = False,
        tau: float = 1e-7,
    ) -> None:
        super().__init__()
        self.register_buffer("sparse_index", torch.arange(d))
        self.features = SparseFeatures
        self.prior = SVIHalfCauchyPrior(d=self.features.num_features, tau=tau)
        self.learn_noise = learn_noise
        self.invsoftplus_sigma = Parameter(inverse_softplus(torch.tensor([noise,])))
        self.num_sparse_features = self.features.num_features

    @property
    def sparse_index(self):
        return self.prior.sparse_index

    def parameters(self):
        if self.learn_noise:
            params = list(self.prior.parameters()) + [
                self.invsoftplus_sigma,
            ]
        else:
            params = self.prior.parameters()
        return params

    def get_natural_parameters(self) -> Tensor:
        return softplus(self.invsoftplus_sigma).pow(2)

    def forward(self, x: Tensor, n: int = 2000) -> Tuple[Tensor, Tensor]:
        w = self.prior.get_reparam_weights(n)
        var = self.get_natural_parameters()
        y_pred = (self.features(x) @ w.T).T
        mu = y_pred.mean(0)
        cov = (y_pred - mu).T @ (y_pred - mu) / (n - 1) + var * torch.eye(x.shape[0])
        return (mu, cov)  # + torch.eye(w.shape[-1]) * var)

    def reparameterized_forward(self, x: Tensor, n: int) -> Tensor:
        w = self.prior.get_reparam_weights(n)
        return w @ self.features(x).T

    def elbo(
        self, x: Tensor, t: Tensor, n_reparams: int, n_data: int, beta: float = 1.0
    ) -> Tensor:
        t_pred = self.reparameterized_forward(x, n_reparams).unsqueeze(-1)
        var = self.get_natural_parameters()
        loglike = isotropic_gaussian_loglikelihood(t.unsqueeze(-1), t_pred, var, 1)
        return loglike - beta * self.prior.kl_divergence() / n_data

    def prune_basis(self) -> None:
        self.prior.update_sparse_index()
        self.features.update_basis(self.sparse_index)
        self.num_sparse_features = int(self.sparse_index.sum())

    def optimize(
        self,
        data_sampler: Callable,
        n_data_total: int,
        lr: float = 1e-2,
        max_iter: int = 30000,
        n_reparams: int = 256,
        beta_warmup_iters: int = 3000,
        print_progress: bool = False,
    ) -> Dict:
        """
        Function that automatically optimizes the sparse glm using SGD. 
        Terminates using "Plfug" diagnostic. This convergence criteria is extremely conservative. 

        Args:
            data_sampler (Callable): function that when called returns a subset of the data inputs and targets
            n_data_total (int): total number of data points in the data set
            lr (float, optional): sgd learning rate. Defaults to 1e-2.
            max_iter (int, optional): Maximum number of SGD iterations. Defaults to 30000.
            n_reparams (int, optional): Number of reparameterization samples. Defaults to 20.

        Returns:
            Dict: Optimization summary, includes convergence boolean, loss history, etc.
        """
        assert (
            beta_warmup_iters < max_iter
        ), "Warm up iterations should be < max iterations."
        if self.learn_noise:
            optimizer = torch.optim.Adam(
                list(self.prior.parameters()) + [self.invsoftplus_sigma,], lr=lr
            )
        else:
            optimizer = torch.optim.Adam(self.prior.parameters(), lr=lr)
        converged = False
        loss_history = []
        message = "Optimizer did not converge before maximum iteration reached."
        S_j = 0
        grad_current = torch.tensor([0.0,])
        for j in range(max_iter):
            grad_previous = grad_current.clone()
            optimizer.zero_grad()
            if beta_warmup_iters == 0:
                beta = 1.0
            else:
                beta = min(1.0, (1.0 * j) / (beta_warmup_iters))
            x, t = data_sampler()
            # computing loss and taking optimization step
            loss = -self.elbo(x, t, n_reparams, n_data_total, beta)
            loss.backward()
            optimizer.step()
            grad_current = torch.cat(
                [
                    p.grad
                    for p_group in optimizer.param_groups
                    for p in p_group["params"]
                ]
            )
            grad_current = grad_current / len(grad_current)
            # checking for convergence
            loss_history.append(loss.item())
            # first ~100 iterations are warm up
            if j > 1:
                S_j = S_j + grad_current @ grad_previous
                if print_progress:
                    if (j + 1) % 1000 == 0:
                        print(
                            "Iter: {:05d} | loss: {:.2f} | S: {:.2f}".format(
                                j + 1, loss.item(), S_j.item()
                            )
                        )
            if len(loss_history) >= 2 and (j > beta_warmup_iters) and (S_j < 0):
                converged = True
                message = "Optimizer converged in %d iterations" % j
                break

        optimization_summary = {
            "loss_history": loss_history,
            "converged": converged,
            "num_iter": j + 1,
            "message": message,
        }
        return optimization_summary


if __name__ == "__main__":
    pass
