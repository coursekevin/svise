import logging
import math
import os
import pathlib
import pickle as pkl
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from functools import partial
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.optim.lr_scheduler as lr_scheduler
from scipy.integrate import solve_ivp
from torch import Tensor, nn
from torch.distributions import MultivariateNormal, kl_divergence
from torch.func import jacfwd, vmap
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from svise import quadrature, sde_learning
from svise.sde_learning import NeuralSDE, SDELearner
from svise.sdeint import solve_sde
from svise.utils import multioutput_gradient

torch.set_default_dtype(torch.float64)

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(CURR_DIR)

from get_data import Data, ExtremeMassBBH, scale, second_order_fd, unscale

MODEL_DIR = os.path.join(CURR_DIR, "results", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

vmapr = partial(vmap, randomness="different")
jacfwdr = partial(jacfwd, randomness="different")


def setup_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    path_to_file = os.path.join(MODEL_DIR, "train.log")
    # Create a file handler
    fh = logging.FileHandler(path_to_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def get_init_state(data: Data) -> Dict[str, Any]:
    t_span = tuple(data.scale_fn(ti) for ti in data.t_span)
    init_state = dict(
        t_span=t_span,
        n_reparam_samples=20,
        drift_layer_description=[128, 128],
        nonlinearity=nn.Tanh(),
        measurement_noise=torch.tensor(1e-2**2),
        tau=1e-5,
        train_t=data.scale_fn(data.train_t),
        n_quad=128,
        quad_percent=0.8,
        n_tau=100,
    )
    return init_state


def set_seeds(rs: int):
    torch.manual_seed(rs)
    np.random.seed(rs)


class BBHUDE(sde_learning.SDEPrior):
    M: float = 1.0
    e: float = 0.5
    p: float = 100
    layer_description: List[int]
    nonlinearity: nn.Module
    zero: Tensor

    def __init__(self, layer_description: List[int], nonlinearity: nn.Module) -> None:
        super().__init__()
        # adding layers
        layers = [(nn.Linear(1, layer_description[0]), nonlinearity)]
        layers += [
            (nn.Linear(layer_description[i], layer_description[i + 1]), nonlinearity)
            for i in range(len(layer_description) - 1)
        ]
        layers = [mod for layer in layers for mod in layer]
        layers.append(nn.Linear(layer_description[-1], 2))
        self.layers = nn.Sequential(*layers)
        self.register_buffer("zero", torch.zeros(1))

    @classmethod
    def base_dynamics(cls, t: Tensor, x: Tensor) -> Tensor:
        _, chi = x.chunk(2, dim=-1)
        const = (1 + cls.e * torch.cos(chi)).pow(2) / (cls.M * (cls.p ** (3 / 2)))
        return const * 6000

    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        _, chi = x.chunk(2, dim=-1)
        fn = self.layers(torch.cos(chi))
        const = self.base_dynamics(t, x)
        dx = const * (1 + fn)
        return dx

    def kl_divergence(self) -> Tensor:
        return self.zero

    def drift(self, t: Tensor, z: Tensor, integration_mode=False) -> Tensor:
        return self.forward(t, z)

    def resample_weights(self) -> None:
        pass


class WaveformLikelihood(sde_learning.Likelihood):
    var: Tensor
    _log2pi: Tensor
    M: float = 1.0
    e: float = 0.5
    p: float = 100

    def __init__(self, measurement_noise: Tensor) -> None:
        super().__init__()
        self.measurement_noise = measurement_noise
        self.register_buffer("var", measurement_noise)
        self.register_buffer("_log2pi", torch.tensor(2 * math.log(2 * math.pi)))

    def pretrain_log_like(
        self, t: Tensor, x: Tensor, marginal_sde: sde_learning.MarginalSDE
    ) -> Tensor:
        raise NotImplementedError

    def second_order_fd(self, f_eval, dt):
        """Finite difference for second order derivative"""
        ddf_eval = torch.zeros_like(f_eval)
        centered_diff = (f_eval[2:] - 2 * f_eval[1:-1] + f_eval[:-2]) / dt**2
        ddf_eval[1:-1] = centered_diff
        # Apply boundary conditions with one-sided difference
        ddf_eval[0] = (
            2 * f_eval[0] - 5 * f_eval[1] + 4 * f_eval[2] - f_eval[3]
        ) / dt**2
        ddf_eval[-1] = (
            2 * f_eval[-1] - 5 * f_eval[-2] + 4 * f_eval[-3] - f_eval[-4]
        ) / dt**2
        return ddf_eval

    def quadrupole(self, zs):
        x, y = zs.chunk(2, dim=-1)
        Ixx = x**2
        Iyy = y**2
        Ixy = x * y
        return torch.cat([Ixx, Iyy, Ixy], dim=-1)

    def waveform(self, n_reparam_samples, marginal_sde, t):
        def qp(t):
            z_samples = marginal_sde.generate_samples(t, n_reparam_samples)
            r_2 = self.convert_to_trajectories(z_samples)
            return self.quadrupole(r_2).squeeze(1)

        ddI = vmapr(func=jacfwdr(func=jacfwdr(func=qp)))(t.unsqueeze(-1)).squeeze()
        ddI = ddI.transpose(1, 0)
        ddIxx, ddIyy, _ = (ddI.div(6000**2)).chunk(3, dim=-1)
        wvf = (ddIxx - ddIyy) * math.sqrt(4 * math.pi / 5)
        return wvf

    def euclidean_norm(self, chi):
        M, e, p = self.M, self.e, self.p
        r = p * M / (1 + e * torch.cos(chi))
        return r

    def convert_to_trajectories(self, state: Tensor) -> Tensor:
        phi, chi = state.chunk(2, dim=-1)
        r = self.euclidean_norm(chi)
        r_2 = torch.cat([r * torch.cos(phi), r * torch.sin(phi)], dim=-1)
        return r_2

    def mean_log_likelihood(
        self,
        t: Tensor,
        x: Tensor,
        marginal_sde: sde_learning.MarginalSDE,
        n_reparam_samples: int,
    ) -> Tensor:
        waveform = self.waveform(n_reparam_samples, marginal_sde, t)
        log_like = -0.5 * (waveform - x).pow(2).div(self.var).sum(-1)
        log_like -= 0.5 * (self.var.log().sum() + self._log2pi * x.shape[-1])
        return log_like.mean()


def setup_sde(
    *,
    n_reparam_samples,
    tau,
    t_span,
    n_tau,
    train_t,
    n_quad,
    quad_percent,
    measurement_noise,
    drift_layer_description,
    nonlinearity,
) -> sde_learning.SDELearner:
    Q_diag = torch.ones(2) * 0.1
    diffusion_prior = sde_learning.SparseDiagonalDiffusionPrior(
        2, Q_diag, n_reparam_samples, tau
    )
    # get an initial guess for the state
    x0 = ExtremeMassBBH().good_initial_condition
    sde_prior = BBHUDE(drift_layer_description, nonlinearity)

    def base_dynamics(t, x):
        dx = BBHUDE.base_dynamics(t, torch.as_tensor(x))
        return dx.detach().numpy()

    soln = solve_ivp(base_dynamics, t_span, x0, atol=1e-8, rtol=1e-6, t_eval=train_t)
    train_y = torch.as_tensor(soln.y.T)

    marginal_sde = sde_learning.DiagonalMarginalSDE(
        2,
        t_span,
        diffusion_prior=diffusion_prior,
        model_form="GLM",
        # kernel="matern12",
        n_tau=n_tau,
        train_x=train_t,
        train_y=train_y,
    )
    # turn off input warping so that we can use jacfwd to estimate ddI
    marginal_sde.m.K.input_warping._update_buffers = False  # type: ignore
    marginal_sde.K_diag.K.input_warping._update_buffers = False  # type: ignore
    likelihood = WaveformLikelihood(measurement_noise)
    quad_scheme = quadrature.UnbiasedGaussLegendreQuad(
        t_span[0], t_span[1], n_quad, quad_percent=quad_percent
    )
    return sde_learning.SDELearner(
        marginal_sde,
        likelihood,
        quad_scheme,
        sde_prior,
        diffusion_prior,
        n_reparam_samples,
    )


@dataclass(frozen=True)
class TrainHParams:
    num_iters: int
    transition_iters: int
    num_mc_samples: int
    lr: float


def initial_cond_kl(x0, S0, nsde: SDELearner):
    mu = nsde.marginal_sde.mean(torch.tensor([0.0])).squeeze(0)
    Sigma = nsde.marginal_sde.K(torch.tensor([0.0])).squeeze(0)
    q = MultivariateNormal(mu, Sigma)
    p = MultivariateNormal(x0, S0)
    return kl_divergence(q, p)


def compute_train_pred_rmse(nsde: SDELearner, data: Data) -> float:
    x0 = data.x0.unsqueeze(0).repeat(nsde.n_reparam_samples, 1)
    t_eval = data.scale_fn(data.train_t)
    sde_kwargs = {"adaptive": True, "atol": 1e-6, "rtol": 1e-4, "dt": 1e-1}
    with torch.no_grad():
        try:
            ys = solve_sde(nsde, x0=x0, t_eval=t_eval, **sde_kwargs)
        except Exception as e:
            return float("inf")
        ys = nsde.likelihood.convert_to_trajectories(ys.transpose(1, 0))  # type: ignore
        med = torch.quantile(ys, 0.5, dim=0)
        x, y = med[:, 0], med[:, 1]
        Ixx = x**2
        Iyy = y**2
        Ixy = x * y
        qp = torch.stack([Ixx, Iyy, Ixy], dim=-1).numpy()
    ddI = second_order_fd(qp, data.train_t[1] - data.train_t[0])
    ddIxx, ddIyy, _ = ddI[..., 0], ddI[..., 1], ddI[..., 2]
    wvf_pred = (ddIxx - ddIyy) * math.sqrt(4 * math.pi / 5)
    rmse = math.sqrt(((wvf_pred - data.train_y.squeeze().numpy()) ** 2).mean())
    return rmse

def train(sde: SDELearner, hparams: TrainHParams, data: Data, logger: logging.Logger):
    assert hparams.transition_iters < hparams.num_iters
    num_data = data.train_y.shape[0]
    summary_freq = 10
    scheduler_freq = hparams.transition_iters // 2
    optimizer = torch.optim.Adam(
        [
            {"params": sde.state_params()},
            {"params": sde.sde_params(), "lr": hparams.lr},
        ],
        lr=hparams.lr,
    )
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    train_dataset = TensorDataset(data.scale_fn(data.train_t), data.train_y)
    batch_size = min(len(train_dataset), hparams.num_mc_samples)
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    num_epochs = hparams.num_iters // len(train_loader)
    j = 0
    loss = torch.tensor([0.0])  # helping type checker
    fname = ""  # helping type checker
    beta = 0.0
    for epoch in range(num_epochs):
        for t_batch, y_batch in tqdm(train_loader):
            j += 1
            beta = min(1.0, j/hparams.transition_iters)
            # if j < 200:
            #     beta = 0.0
            # else:
            #     beta = min(1.0, ((j - 200) / 5000) ** 2)

            if j % scheduler_freq == 0:
                scheduler.step()

            optimizer.zero_grad()
            loss = -sde.elbo(t_batch, y_batch, beta, num_data, compat_mode=False)
            S0 = torch.eye(2) * (1e-2**2)
            kl_init = initial_cond_kl(data.x0, S0, sde)
            loss += kl_init * beta
            loss.backward()
            optimizer.step()

        fname = f"nsde-{epoch:05d}-{j:05d}.pt"
        if epoch % summary_freq == 0:
            sde.eval()
            log_msg = f"Epoch {epoch:04d} | beta: {beta:.2e} | Loss {loss.item():.2e}"
            logger.info(log_msg)
            torch.save(sde.state_dict(), os.path.join(MODEL_DIR, fname))
            subprocess.Popen(
                ["python", os.path.join(CURR_DIR, "post_process.py")], cwd=CURR_DIR
            )
            sde.train()
    sde.eval()
    return fname


def main():
    set_seeds(23)
    logger = setup_logger()
    data = Data.load(CURR_DIR)
    # get init state and freeze
    init_state = get_init_state(data)
    with open(os.path.join(MODEL_DIR, "init_state_freeze.pkl"), "wb") as f:
        pkl.dump(init_state, f)
    nsde = setup_sde(**init_state)
    nsde.train()
    hparams = TrainHParams(
        num_iters=int(2 * 50000), transition_iters=1000, num_mc_samples=256, lr=1e-2
    )
    with open(os.path.join(MODEL_DIR, "train_hparams.pkl"), "wb") as f:
        pkl.dump(asdict(hparams), f)
    # training model
    final_fname = train(nsde, hparams, data, logger)
    torch.save(nsde.state_dict(), os.path.join(MODEL_DIR, final_fname))
    print("Done.")


if __name__ == "__main__":
    main()
