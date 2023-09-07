"""
Script for training a neural SDE on cylinder flow POD modes.
"""
import os
import sys
import pathlib
import pickle as pkl
from typing import Any
import logging
from functools import wraps
from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from svise.sde_learning import NeuralSDE

CURR_DIR = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(CURR_DIR)
from generate_data import DATA_DIR

MODEL_DIR = os.path.join(CURR_DIR, "results", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

torch.set_default_dtype(torch.float64)

def set_seeds(rs: int):
    torch.manual_seed(rs)
    np.random.seed(rs)

@dataclass(frozen=True)
class NSDEHParams:
    n_reparam_samples: int
    drift_layer_description: list[int]
    nonlinearity: nn.Module
    tau: float
    n_quad: int
    quad_percent: float
    n_tau: int

@dataclass(frozen=True)
class TrainHParams:
    num_iters: int
    transition_iters: int
    batch_size: int
    summary_freq: int
    lr: float

def get_hparams() -> tuple[NSDEHParams, TrainHParams]:
    sde_hparams = NSDEHParams(
    n_reparam_samples = 32,
    drift_layer_description = [128, ],
    nonlinearity = nn.Tanh(),
    tau = 1e-5,
    n_quad = 200,
    quad_percent = 0.5,
    n_tau = 500)
    train_hparams = TrainHParams(
    num_iters = int(20000),
    transition_iters = 5000,
    batch_size = 64,
    summary_freq = 100,
    lr = 1e-3)
    return sde_hparams, train_hparams

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

def get_data() -> dict[str, Any]:
    with open(os.path.join(DATA_DIR, "encoded_data.pkl"), "rb") as handle:
        data = pkl.load(handle)
    dtype = torch.float64
    return dict(num_data = data["z"].shape[0],
                d = data["z"].shape[1],
                t = data["t"].to(dtype),
                y_data = data["z"].to(dtype),
                var = data["code_stdev"].pow(2).to(dtype),
                t_span = (float(data["t"].min()), float(data["t"].max())))

def get_init_state(data: dict[str, Any], sde_hparams: NSDEHParams) -> dict[str, Any]:
    data_init_state = dict(d=data["d"],
                           t_span=data["t_span"], 
                           G=torch.eye(data["d"]),
                           measurement_noise=data["var"], 
                           train_t=data["t"], 
                           train_x=data["y_data"])
    return {**data_init_state, **asdict(sde_hparams)}

def initialize_model(init_state: dict[str, Any]) -> NeuralSDE:
    nsde = NeuralSDE(**init_state)
    nsde.train()
    return nsde

def no_grad(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper

@no_grad
def plot_summary(curr_iter: int, device: torch.device, nsde: NeuralSDE, data: dict[str, Any]):
    plot_dir = os.path.join(MODEL_DIR, "plots", f"summary_{curr_iter}")
    os.makedirs(plot_dir, exist_ok=True)
    nsamples = nsde.n_reparam_samples
    ys_train = nsde.marginal_sde.generate_samples(data["t"].to(device), nsamples)

    median = torch.quantile(ys_train, 0.5, dim=0)
    n_plots = median.shape[1]
    max_plots_per_fig = 10
    n_figs = n_plots // max_plots_per_fig + (n_plots % max_plots_per_fig > 0)

    for i in range(n_figs):
        n_plots_in_this_fig = min(max_plots_per_fig, n_plots-i*max_plots_per_fig)
        fig, axes = plt.subplots(n_plots_in_this_fig, 1, figsize=(24, 3))
        for j in range(n_plots_in_this_fig):
            mode_idx = i*max_plots_per_fig+j
            if mode_idx < n_plots:  # Check if index is within range
                axes[j].plot(data["t"], data["y_data"][:, mode_idx])
                axes[j].plot(data["t"], median[:, mode_idx].cpu())
        plot_name = os.path.join(plot_dir, f"pod_median_{i}.png")
        fig.savefig(plot_name)
        plt.close(fig)

def cuda(device, iterable):
    for x in iterable:
        yield (xi.to(device) for xi in x)

def save_checkpoint(curr_iter: int, nsde: NeuralSDE):
    ckpt_path = os.path.join(MODEL_DIR, f"nsde_{curr_iter:06d}.pt")
    torch.save(nsde.state_dict(), ckpt_path)

def train_step(nsde: NeuralSDE, optimizer: torch.optim.Optimizer, batch: tuple[Tensor, Tensor], beta: float, num_data:int):
    t_batch, y_batch = batch
    optimizer.zero_grad()
    loss = -nsde.elbo(t_batch, y_batch, beta, num_data, compat_mode=False)
    loss.backward()
    optimizer.step()
    return loss

def train(nsde: NeuralSDE, hparams: TrainHParams, data: dict[str, Any], device: torch.device):
    nsde.to(device)
    assert hparams.transition_iters < hparams.num_iters
    logger = logging.getLogger(__name__)
    summary_freq = 1000
    scheduler_freq = hparams.transition_iters // 2
    optimizer = torch.optim.Adam(
        [
            {"params": nsde.state_params()},
            {"params": nsde.sde_params(), "lr": hparams.lr},
        ],
        lr=hparams.lr / 10,
    )
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    train_dataset = TensorDataset(data["t"], data["y_data"])
    train_loader = DataLoader(train_dataset, batch_size=hparams.batch_size, shuffle=True)
    num_epochs = hparams.num_iters // len(train_loader)
    j = 0
    for _ in range(num_epochs):
        for t_batch, y_batch in cuda(device, train_loader):
            j += 1
            beta = min(1.0, (1.0 * j) / (hparams.transition_iters))
            if j % scheduler_freq == 0:
                scheduler.step()
            loss = train_step(nsde, optimizer, (t_batch, y_batch), beta, data["num_data"])
            if j % summary_freq == 0:
                nsde.eval()
                log_msg = f"Iter {j:06d} | beta: {beta:.2e} | loss {loss.item():.2e}"
                logger.info(log_msg)
                save_checkpoint(j, nsde)
                nsde.train()
    save_checkpoint(j, nsde)
            

def main():
    set_seeds(23)
    logger = setup_logger()
    data = get_data()
    sde_hparams, train_hparams = get_hparams()
    init_state = get_init_state(data, sde_hparams)
    with open(os.path.join(MODEL_DIR,"init_state_freeze.pkl"), "wb") as handle:
        pkl.dump(init_state, handle)
    nsde = initialize_model(init_state)
    train(nsde, train_hparams, data, torch.device("cuda:0"))
    logger.info("Done.")
    print("Done.")


if __name__ == "__main__":
    main()