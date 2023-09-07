import torch
import gpytorch
import torch.nn as nn
from botorch.fit import fit_gpytorch_model
import numpy as np


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# initialize likelihood and model
class SingleTaskGP(nn.Module):
    def __init__(self, train_x, train_y, noise=None) -> None:
        super().__init__()
        with torch.enable_grad():
            with gpytorch.settings.fast_computations(
                covar_root_decomposition=False, log_prob=False, solves=False
            ):
                if noise is None:
                    self.noise = None
                    self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
                else:
                    self.noise = noise
                    self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
                        noise * torch.ones(len(train_x))
                    )
                self.model = ExactGPModel(train_x, train_y, self.likelihood)
                self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                    self.likelihood, self.model
                )
                out = fit_gpytorch_model(self.mll)

    def forward(self, x, full_covariance=True):
        with gpytorch.settings.fast_computations(
            covar_root_decomposition=False, log_prob=False, solves=False
        ):
            out = self.model(x)
        if full_covariance:
            return out.mean, out.covariance_matrix
        else:
            return out.mean, out.variance


class MultitaskGP(nn.Module):
    def __init__(self, train_x, train_y, var) -> None:
        super().__init__()
        assert train_y.shape[1] == len(var), "variance must be provided for each output"
        num_outputs = train_y.shape[1]
        if len(train_x) > 500:
            idx = np.arange(len(train_x))
            np.random.shuffle(idx)
            train_x = train_x[idx[:500]]
            train_y = train_y[idx[:500]]
        self.gps = nn.ModuleList(
            [
                SingleTaskGP(train_x, train_y[:, i], noise=var[i])
                for i in range(num_outputs)
            ]
        )

    def forward(self, x):
        means = []
        covs = []
        for gp in self.gps:
            m, var = gp(x, full_covariance=False)
            means.append(m)
            covs.append(var)
        means = torch.stack(means, dim=-1)
        stds = torch.stack(covs, dim=-1).sqrt()
        return means, stds

