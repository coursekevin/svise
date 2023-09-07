from abc import ABC, abstractmethod
import math
from typing import Callable, Tuple, Union
from typing import Callable

from gpytorch import kernels
import numpy as np
import scipy.linalg
import torch
from torch import sigmoid
from torch import Tensor
from torch.autograd import backward
import torch.nn as nn
from torch.nn.parameter import Parameter

# from .extern import uwbayes


def finite_difference(func, t, dt):
    """
    Computes the finite difference approximation of the derivative of func at t.
    """
    return (func(t + dt) - func(t - dt)) / (2 * dt)


def skew_sym_duplication_matrix(n: int) -> Tensor:
    """
    Creates the duplication matrix for the skew symmetric matrix of size n
    ie. Dn @ v(A) = vec(A)
    """
    D = torch.zeros(n * (n - 1) // 2, n**2)
    k = 0
    for j in range(n):
        for i in range(n):
            if i > j:
                u = torch.zeros(n * (n - 1) // 2)
                u[k] = 1.0
                T = torch.zeros(n, n)
                T[i, j] = 1.0
                T[j, i] = -1.0
                D += u.unsqueeze(-1) * T.t().ravel().unsqueeze(0)
                k += 1
    return D.t()


def sample_covariance(x: Tensor) -> Tensor:
    """
    Computes the sample covariance of x
    Args:
        x (Tensor): (...,bs, d) batch of inputs
    Returns:
        Tensor: (..., d, d) sample covariance
    """
    N = x.shape[-2]
    mu = x.mean(-2).unsqueeze(-2)
    diff = x - mu
    return 1 / (N - 1) * diff.transpose(-1, -2) @ diff


def make_random_matrix(dim: int, rank: int, random_seed: int = None) -> Tensor:
    """
    Generates a random low rank matrix

    Args:
        dim (int): dimension of matrix
        rank (int): rank of matrix
        random_seed (int, optional): random seed of matrix. Defaults to None.

    Returns:
        Tensor: matrix of rank and dim
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)
    assert dim >= rank, "rank cannot be greater than dim."
    u = torch.randn(rank, dim, 1)
    return (u @ u.transpose(-2, -1)).sum(0).div(rank)


def bjorck(
    A: Tensor,
    tol: float = 1e-9,
    max_iters: int = 50,
    order: int = 1,
    num_iters=None,
) -> Tensor:
    """
    Finds closest orthonormal matrix to A via Bjorck orthonomralization

    Args:
        A (Tensor): Input tensor
        tol (float, optional): Tolerance on ||O - I||_F^2. Defaults to 1e-9.
        max_iters (int, optional): Maximum iterations. Defaults to 50.
        order (int, optional): Order of iteration. Defaults to 1.

    Returns:
        Tensor: Orthonormal matrix closest to A
    """

    N, M = A.shape[-2:]
    scale = math.sqrt(N * M)  # safe bjorck scaling
    I = torch.eye(M)
    Ak = A / scale
    for j in range(max_iters):
        Qk = I - Ak.transpose(-2, -1) @ Ak
        if num_iters is None:
            if Qk.pow(2).sum() < tol:
                break  # check sqr frob. norm is less than tol
        elif j == num_iters:
            break
        if order == 1:
            Ak = Ak @ (I + 1 / 2 * Qk)
        elif order == 2:
            Ak = Ak @ (I + 1 / 2 * Qk + 3 / 8 * (Qk @ Qk))
        else:
            raise ValueError(f"Order: {order} not supported.")
    return Ak, j


def solve_least_squares(A: Tensor, B: Tensor, gamma: float = 1e-6) -> Tensor:
    """
    Solves a least squares problem using SVD for stability.

    Args:
        A (Tensor): (n, m) design matrix
        B (Tensor): (n, d) right hand side

    Returns:
        Tensor: W where A @ W = B
    """
    assert A.dim() == 2, "A must be a matrix"
    assert B.dim() == 2, "B must be a matrix"
    U, S, Vh = torch.linalg.svd(A, full_matrices=False)
    return Vh.t() @ (U.t() @ B).mul((S / (S.pow(2) + gamma)).unsqueeze(-1))


class AffineTransform(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Computes x_tilde(t) = A x(t) + b

        Args:
            x (Tensor): (..., d) batch of inputs

        Returns:
            Tensor: (..., d) batch of outputs
        """
        pass

    @abstractmethod
    def transform(self, x: Tensor) -> Tensor:
        """
        Computes x_tilde(t) = A x(t) + b

        Args:
            x (Tensor): (..., d) batch of inputs

        Returns:
            Tensor: (..., d) batch of outputs
        """
        return self.forward(x)

    @abstractmethod
    def inverse(self, x: Tensor) -> Tensor:
        """
        Computes x(t) = A^-1 (x_tilde(t) - b)

        Args:
            x (Tensor): (..., d) batch of inputs

        Returns:
            Tensor: (..., d) batch of outputs
        """
        pass

    @abstractmethod
    def scale(self, dx: Tensor) -> Tensor:
        """
        Computes x_tilde(t) = A x(t). Useful for computing derivative of transform.

        Args:
            dx (Tensor): (..., d) batch of inputs

        Returns:
            Tensor: (..., d) batch of outputs
        """
        pass


class StandardizeTransform(AffineTransform):
    def __init__(self, x: Tensor):
        super().__init__()
        # assert len(x.shape) == 2
        self.register_buffer("mu", x.mean(0))
        self.register_buffer("sigma", x.std(0))

    def forward(self, x: Tensor) -> Tensor:
        return (x - self.mu) / self.sigma

    def transform(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def inverse(self, x: Tensor) -> Tensor:
        return x * self.sigma + self.mu

    def scale(self, dx: Tensor) -> Tensor:
        return dx / self.sigma


class ScaleTransform(nn.Module, ABC):
    """
    Transformation for scaling data by a factor over each dimension
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    @abstractmethod
    def scale(self) -> Tensor:
        """
        Scaling factor

        Returns:
            Tensor: (d,) scaling factor
        """
        pass

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        In general computes x / self.scale

        Args:
            x (Tensor): input

        Returns:
            Tensor: scales x by 1 / self.scale
        """
        pass

    @abstractmethod
    def inverse(self, x: Tensor) -> Tensor:
        """
        Computes the inverse of the transform, in general x * self.scale

        Args:
            x (Tensor): input

        Returns:
            Tensor: scales x by self.scale
        """
        pass


class StdevScaleTransform(ScaleTransform):
    def __init__(self, x: Tensor):
        super().__init__()
        self.register_buffer("sigma", x.std(0))

    @property
    def scale(self) -> Tensor:
        return self.sigma

    def forward(self, x: Tensor) -> Tensor:
        return x / self.scale

    def inverse(self, x: Tensor) -> Tensor:
        return x * self.scale


class IdentityScaleTransform(ScaleTransform):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.register_buffer("ones", torch.ones(d))

    @property
    def scale(self) -> Tensor:
        return self.ones

    def forward(self, x: Tensor) -> Tensor:
        return x

    def inverse(self, x: Tensor) -> Tensor:
        return x


class MultivariateStandardizeTransform(nn.Module):
    def __init__(self, x: Tensor, R: Tensor = None):
        super().__init__()
        self.register_buffer("mu", x.mean(0))
        if R is None:
            R = sample_covariance(x)
        self.register_buffer("L", torch.linalg.cholesky(R))

    def forward(self, x: Tensor) -> Tensor:
        return torch.triangular_solve(
            (x - self.mu).t(), self.L, upper=False
        ).solution.t()

    def inverse(self, x: Tensor) -> Tensor:
        return x @ self.L.t() + self.mu


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def inverse(self, x):
        return x


class Positive(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.output_transform = nn.Softplus(beta=beta)
        self.minimum = 1e-6

    def forward(self, x, return_grad=False):
        res = self.output_transform(x) + self.minimum
        if not return_grad:
            return res
        else:
            grad_res = sigmoid(x * self.output_transform.beta)
            return res, grad_res


class ReHU(nn.Module):
    """Rectified Huber unit
    from: https://github.com/locuslab/stable_dynamics/blob/master/models/stabledynamics.py
    """

    def __init__(self, d: float = 1.0):
        super(ReHU, self).__init__()
        self.a = 1 / d
        self.b = -d / 2

    def forward(self, x):
        return torch.max(
            torch.clamp(torch.sign(x) * self.a / 2 * x**2, min=0, max=-self.b),
            x + self.b,
        )


class GLM(nn.Module):
    def __init__(
        self,
        output_size: int,
        a: float,
        b: float,
        n_tau: int,
        kernel: str = "matern52",
        learn_inducing_locations: bool = False,
        output_transform: Callable = Identity(),
    ) -> None:
        super().__init__()
        self.w = Parameter(torch.randn(n_tau, output_size))
        tau = torch.linspace(a, b, n_tau)
        if learn_inducing_locations:
            self.tau = Parameter(tau)
        else:
            self.register_buffer("tau", tau)
        if kernel == "matern52":
            self.K = kernels.MaternKernel(nu=2.5)
        elif kernel == "matern32":
            self.K = kernels.MaternKernel(nu=1.5)
        elif kernel == "rbf":
            self.K = kernels.RBFKernel()
        else:
            raise ValueError(f"Unknown kernel {kernel}")
        self.w = Parameter(
            torch.linalg.cholesky(
                self.K(tau, tau).evaluate().inverse() + 1e-4 * torch.eye(len(tau))
            )
            @ torch.randn(n_tau, output_size)
        )
        self.b = Parameter(torch.zeros(output_size))
        self.output_transform = output_transform

    def forward(self, t: Tensor) -> Tensor:
        if t.dim() == 0:
            t = t.unsqueeze(-1)
        return self.output_transform(self.K(t, self.tau).evaluate() @ self.w + self.b)


class FCNN(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        nonlinearity: Callable = nn.Softplus(),
        output_transform: Callable = Identity(),
        batch_norm: bool = True,
    ) -> None:
        """
        Fully connected neural network with batchnorm

        Args:
            input_size (int): input dimension
            hidden_size (int): number of hidden units in each hidden layer
            output_size (int): output dimension
            num_layers (int): number of hidden layers
            nonlinearity (Callable, optional): nonlinearity at each layer. Defaults to nn.Softplus().
        """
        super(FCNN, self).__init__()
        if batch_norm:
            layers = [
                nn.Linear(input_size, hidden_size),
                nonlinearity,
                nn.BatchNorm1d(hidden_size),
            ]
            for j in range(num_layers - 1):
                layers += [
                    nn.Linear(hidden_size, hidden_size, bias=True),
                    nonlinearity,
                    nn.BatchNorm1d(hidden_size),
                ]
        else:
            layers = [
                nn.Linear(input_size, hidden_size),
                nonlinearity,
            ]
            for j in range(num_layers - 1):
                layers += [
                    nn.Linear(hidden_size, hidden_size),
                    nonlinearity,
                ]
        layers += [nn.Linear(hidden_size, output_size), output_transform]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 0:
            return self.mlp(x.reshape(1, 1))
        if x.dim() == 1:
            return self.mlp(x.unsqueeze(-1))
        elif x.dim() == 2:
            return self.mlp(x)
        else:
            bs = x.shape[:-1]
            d = x.shape[-1]
            return self.mlp(x.reshape(-1, d)).reshape(*bs, -1)


def multioutput_gradient(f: Tensor, t: Tensor, vmap=True) -> Tensor:
    """
    Computes the gradient of a f at a batch of times t.

    Args:
        f (Tensor): (bs,d) tensor of function evaluations
        t (Tensor): (bs,) batch of times
        vmap (bool, optional): Bool indicating whether to use vmap. Defaults to True.

    Returns:
        Tensor: (bs,d) gradient of f at t
    """
    d = f.shape[-1]
    basis_vectors = torch.eye(d)
    # looped solution
    if not vmap:
        jacobian_rows = [
            torch.autograd.grad(f.sum(0), t, v, retain_graph=True)[0]
            for v in basis_vectors.unbind()
        ]
        jacobian = torch.stack(jacobian_rows, dim=-1)
    # experiemental, requires torch nightly build
    else:
        torch._C._debug_only_display_vmap_fallback_warnings(True)

        def get_vjp(v):
            return torch.autograd.grad(f.sum(0), t, v, retain_graph=True)[0]

        jacobian = torch.vmap(get_vjp)(basis_vectors).t()
    return jacobian


class EulerMaruyama(object):
    """
    Performs euler-maruyama integration of an SDE

    Args:
        f (Callable): (float, (bs, d)) -> (bs,d) drift function computes a batch of drift values
        L (Union[Callable, Tensor]): Union[(float, (bs, d)) -> (bs, d, S), (d,S)] diffusion function computes a batch of diffusion values
        Q (Tensor): (S,S) diffusion matrix of Brownwian motion
        t_span (Tuple): integration time window
        x0 (Tensor): (bs,d) initial batch of samples from initial condition
        dt (float): integration time step
    """

    def __init__(
        self,
        f: Callable,
        L: Union[Callable, Tensor],
        Q: Tensor,
        t_span: Tuple,
        x0: Tensor,
        dt: float,
    ) -> None:
        super().__init__()
        bs = x0.shape[0]
        Q_chol = torch.linalg.cholesky(Q) * math.sqrt(dt)
        t = torch.arange(t_span[0], t_span[1] + dt, dt)
        xk = x0.clone()
        x_list = [xk]
        for tk in t[:-1]:
            dbeta = torch.randn(bs, Q_chol.shape[0]) @ Q_chol.t()
            xk = xk + f(tk, xk) * dt
            if callable(L):
                noise_samples = torch.einsum("ijk,ik->ij", L(tk, xk), dbeta)
            else:
                noise_samples = dbeta @ L.t()
            xk += noise_samples
            x_list.append(xk.clone())
        self.x = torch.stack(x_list, dim=1)
        if x0.shape[0] > 1:
            self.mean = self.x.mean(0)
            self.var = sample_covariance(self.x.transpose(1, 0))
        self.t = t


def solve_lyapunov_spectral(D: Tensor, Q: Tensor, RHS: Tensor) -> Tensor:
    """
    Solves the lyapunov equation, XK^T + KX^T = RHS, using the spectral decomposition of K

    Args:
        D (Tensor): (..., n) array of eigen values
        Q (Tensor): (n,n)
        RHS (Tensor): (n,n) right hand side

    Returns:
        Tensor: (n,n) solution
    """
    n = D.shape[-1]
    if D.dim() > 1:
        bs = D.shape[:-1]
        squeeze = False
    else:
        bs = (1,)
        squeeze = True
    bs_rhs = RHS.shape[:-2]
    newRHS = Q.transpose(-2, -1) @ RHS @ Q
    # generate all eigenvalues of K \oplus K
    eigs = (D.unsqueeze(-1) + D.unsqueeze(-2)).reshape(*bs, -1)
    # eigs = torch.kron(D, torch.ones(n)) + torch.kron(torch.ones(n), D)
    vecX = newRHS.transpose(-2, -1).reshape(*bs_rhs, -1) / eigs
    X = Q @ vecX.reshape(*bs_rhs, n, n).transpose(-2, -1) @ Q.transpose(-2, -1)
    if squeeze:
        X = X.squeeze()
    return X


def solve_lyapunov_diag(K_diag: Tensor, RHS_diag: Tensor) -> Tensor:
    """
    Solves the lyapunov equation XK^T + KX^T = RHS in the case that
    K and RHS are diagonal (order(N) operations)

    Args:
        K_diag (Tensor): (n,) diagonal of K
        RHS_diag (Tensor): (n,) diagonal of RHS

    Returns:
        Tensor: (n,n) matrix solution
    """
    # batch-wise case for SVI
    if RHS_diag.dim() == 3:
        pass
    elif K_diag.shape[0] != RHS_diag.shape[0]:
        # todo: what case was this for?
        RHS_diag = RHS_diag.unsqueeze(1)
    return 0.5 * RHS_diag / K_diag


def mean_diagonal_gaussian_loglikelihood(
    x: Tensor, mu: Tensor, var: Tensor, log2pi: Tensor = None
) -> Tensor:
    """
    Computes the mean log likelihood for a batch of data points given a batch of means and variances

    Args:
        x (Tensor): (bs, d) batch of data points
        mu (Tensor): (ns, bs, d) batch of means
        var (Tensor): (d,) variances

    Returns:
        Tensor: mean gaussian log likelihood
    """
    k = x.shape[-1]
    if log2pi is None:
        log2pi = torch.log(torch.tensor(2 * math.pi, dtype=x.dtype, device=x.device))
    diff = ((x - mu).pow(2) / var).sum(-1)
    return -0.5 * diff.mean() - k / 2 * log2pi - 0.5 * torch.log(var.prod())


def grad_skew_expm(V, exp_D, V_trns_conj, D, triu_indices, d_indices, dupl_mat):
    # https://www.janmagnus.nl/wips/expo-23.pdf
    d = len(d_indices)
    Delta = torch.zeros_like(V)
    diff = (exp_D[..., triu_indices[0]] - exp_D[..., triu_indices[1]]) / (
        D[..., triu_indices[0]] - D[..., triu_indices[1]]
    )
    Delta[..., triu_indices[0], triu_indices[1]] = diff
    Delta[..., triu_indices[1], triu_indices[0]] = diff
    Delta[..., d_indices, d_indices] = exp_D
    if V.dim() == 2:
        R = torch.kron(V.conj(), V)
        R_inv_dupl = torch.kron(V.transpose(-2, -1), V_trns_conj) @ dupl_mat
    elif V.dim() == 3:
        # batch wise kronecker product
        R = torch.einsum("ijk,ilm->ijlkm", V.conj(), V).view(-1, d**2, d**2)
        R_inv_dupl = (
            torch.einsum("ijk,ilm->ijlkm", V.transpose(-2, -1), V_trns_conj).view(
                -1, d**2, d**2
            )
            @ dupl_mat
        )
    else:
        raise ValueError("S must be 2 or 3 dimensional")
    grad_exp_S = (
        R * Delta.flatten(start_dim=-2, end_dim=-1).unsqueeze(-2) @ (R_inv_dupl)
    )
    return grad_exp_S.real


class FastSkewSymMatrixExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, S_vec, triu_indices, d_indicies, dupl_mat):
        d = len(d_indicies)
        S = torch.zeros((*S_vec.shape[:-1], d, d))
        S[..., triu_indices[0], triu_indices[1]] = -S_vec
        S[..., triu_indices[1], triu_indices[0]] = S_vec
        D, V = torch.linalg.eigh(1.0j * S)  # j * S is Hermitian
        D = D / 1.0j
        exp_D = torch.exp(D)
        V_trns_conj = V.transpose(-2, -1).conj()
        ctx.save_for_backward(
            V, exp_D, V_trns_conj, D, triu_indices, d_indicies, dupl_mat
        )
        return (V * exp_D.unsqueeze(-2) @ V_trns_conj).real

    @staticmethod
    def backward(ctx, gradoutputs):
        bs = gradoutputs.shape[:-2]
        grad = (
            gradoutputs.transpose(-2, -1).reshape(*bs, 1, -1)
            @ grad_skew_expm(*ctx.saved_tensors)
        ).squeeze(-2)
        return (grad, None, None, None, None)


fast_skew_sym_matrix_exp = FastSkewSymMatrixExp.apply


class SkewSymMatrixExp(nn.Module):
    """
    This module computes the matrix exponential of a skew symmetric matrix along with its gradient
    In testing this is about 10x-100x faster than autograd
    """

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.register_buffer("dupl_mat", skew_sym_duplication_matrix(d) + 0j)
        self.register_buffer("ind", torch.triu_indices(d, d, offset=+1))
        self.register_buffer("d_ind", torch.arange(d))

    def forward(self, S_vec: Tensor, return_grad: bool = False):
        """Computes matrix exp given n(n-1)/2 input vector
            self ([TODO:parameter]): [TODO:description]
            S_vec (Tensor): batch of flattened n(n-1)/2 Tensor defining skew sym matrices
            return_grad (bool): flag indicating whether to return output or output and grad

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]: exp(S_vec) or (exp(S_vec), grad(exp(S_vec)))
        """
        exp_S = fast_skew_sym_matrix_exp(S_vec, self.ind, self.d_ind, self.dupl_mat)
        if not return_grad:
            return exp_S
        else:
            grad_exp_S = grad_skew_expm(*exp_S.grad_fn.saved_tensors)
            return exp_S, grad_exp_S


# ------------------------------------------------------------------------------
# logm from https://github.com/pytorch/pytorch/issues/9983
# ------------------------------------------------------------------------------


def adjoint(A, E, f):
    A_H = A.T.conj().to(E.dtype)
    n = A.size(0)
    M = torch.zeros(2 * n, 2 * n, dtype=E.dtype, device=E.device)
    M[:n, :n] = A_H
    M[n:, n:] = A_H
    M[:n, n:] = E
    return f(M)[:n, n:].to(A.dtype)


def logm_scipy(A):
    return torch.from_numpy(scipy.linalg.logm(A.cpu(), disp=False)[0]).to(A.device)


class Logm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A):
        assert A.ndim == 2 and A.size(0) == A.size(1)  # Square matrix
        assert A.dtype in (
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        )
        ctx.save_for_backward(A)
        return logm_scipy(A)

    @staticmethod
    def backward(ctx, G):
        (A,) = ctx.saved_tensors
        return adjoint(A, G, logm_scipy)


logm = Logm.apply


def matrix_log(A: Tensor) -> Tensor:
    if A.dim() == 3:
        return torch.stack([logm(Ai) for Ai in A], dim=0)
    else:
        return logm(A)
