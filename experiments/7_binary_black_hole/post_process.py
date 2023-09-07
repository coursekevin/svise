import os
import math
import re
import sys
import pathlib
import pickle as pkl
from typing import List, Tuple
from functools import wraps

import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from svise.sde_learning import NeuralSDE, SDELearner
from svise.sdeint import solve_sde

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
sys.path.append(CURR_DIR)
FIG_DIR = os.path.join(CURR_DIR, "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

from get_data import Data, scale, unscale, ExtremeMassBBH, second_order_fd
from experiment_util import setup_sde, MODEL_DIR

FILL_COLOR = "#92c5de"
TRAIN_STYLE = dict(color="k", linestyle="-.", alpha=1.0, label="Training", linewidth=0.8)
VALID_STYLE = dict(
    color="#fc8d62", linestyle="-.", alpha=1.0, label="Validation", linewidth=0.8
)
PRED_STYLE = dict(
    color="C0", linestyle="-", alpha=1.0, label="Median Pred.", linewidth=0.8
)

sns.set(style="ticks")
matplotlib.rc("text", usetex=True)
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.preamble'] = r'\usepackage{helvet} \usepackage{sansmath} \sansmath'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.labelsize'] = 7
plt.rcParams['xtick.labelsize'] = 5
plt.rcParams['ytick.labelsize'] = 5
WIDTH_INCHES = 89 / 25.4 # Nature column width

def convert_width(fsize: tuple[float, float]) -> tuple[float, float]:
    width = fsize[0] 
    return tuple(size * WIDTH_INCHES / width for size in fsize)


def no_grad(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper

def predict(nsde: SDELearner, data: Data, nsamples: int) -> Tensor:
    sde_kwargs = {"adaptive": True, "atol": 1e-6, "rtol": 1e-4, "dt": 1e-2}
    y0 = data.valid_x0.unsqueeze(0).repeat(nsamples, 1)
    t_eval = data.scale_fn(data.valid_t)
    ys = solve_sde(nsde, x0=y0, t_eval=t_eval, **sde_kwargs)
    ys = nsde.likelihood.convert_to_trajectories(ys.transpose(1, 0))  #  # type: ignore
    return ys


def parse_log_file(filename: str) -> List[Tuple[float, int, float, float]]:
    with open(filename, "r") as file:
        log_data = []
        for line in file:
            epoch = int(line.split(" ")[4])
            beta = float(line.split(" ")[7])
            loss = float(line.split(" ")[10].strip("\n"))
            log_data.append((epoch, beta, loss))
    log_data = sorted(log_data)
    return log_data

def figname_suffix(epoch_iter: tuple[int, int]) -> str:
    epoch, j = epoch_iter
    return f"-{epoch:05d}-{j:05d}.pt"

def get_latest_model(nsamples: int) -> tuple[SDELearner, tuple[int, int]]:
    with open(os.path.join(MODEL_DIR, "init_state_freeze.pkl"), "rb") as f:
        init_state = pkl.load(f)
    init_state["n_reparam_samples"] = nsamples
    nsde = setup_sde(**init_state)
    models = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")])
    final_state_dict = models[-1]
    epoch_iter = tuple(int(i) for i in final_state_dict.removesuffix(".pt").split("-")[1:])
    print("Loading model: ", final_state_dict)
    nsde.load_state_dict(torch.load(os.path.join(MODEL_DIR, final_state_dict)))
    nsde.resample_sde_params()
    nsde.eval()
    return nsde, epoch_iter


def get_soln_samples(
    nsde: SDELearner, data: Data, nsamples: int
) -> tuple[Tensor, Tensor]:
    # validation time window
    ys = nsde.marginal_sde.generate_samples(data.scale_fn(data.train_t), nsamples) # type: ignore
    ys = nsde.likelihood.convert_to_trajectories(ys) # type: ignore
    ys = torch.cat([ys, predict(nsde, data, nsamples)], dim=1)
    t_eval = torch.cat([data.train_t, data.valid_t])
    return t_eval, ys

def convert_to_polar(traj: Tensor) -> tuple[Tensor, Tensor]:
    theta = torch.atan2(traj[..., 1], traj[..., 0])
    r = torch.norm(traj, dim=-1)
    return theta, r

@no_grad
def plot_state_estimates(nsde: SDELearner, data: Data, figsuffix: str):
    ys = nsde.marginal_sde.generate_samples(data.scale_fn(data.train_t), 1024)
    ys = nsde.likelihood.convert_to_trajectories(ys) # type: ignore
    mean = torch.mean(ys, dim=0)
    std = torch.std(ys, dim=0)
    fig, ax = plt.subplots(3, 1, figsize=convert_width((6, 4)), sharex=True)
    for j in range(2):
        ax[j].plot(data.train_t, data.train_r[:, j], **TRAIN_STYLE)
        ax[j].plot(data.train_t, mean[:, j], **PRED_STYLE)
        ax[j].fill_between(
            data.train_t,
            mean[:, j] - std[:, j],
            mean[:, j] + std[:, j],
            alpha=0.2,
            color=FILL_COLOR,
        )
    wvf_pred = nsde.likelihood.waveform(1024, nsde.marginal_sde, data.scale_fn(data.train_t)) # type: ignore
    ax[2].plot(data.train_t, data.train_y, **TRAIN_STYLE)
    ax[2].plot(data.train_t, torch.mean(wvf_pred, 0), **PRED_STYLE)
    fig.savefig(os.path.join(FIG_DIR, f"state-est{figsuffix}.pdf"), format="pdf", bbox_inches="tight")

@no_grad
def get_pred_plot(soln_sample: tuple[Tensor, Tensor], data: Data, figsuffix: str):
    t_eval, ys = soln_sample
    median = torch.quantile(ys, 0.5, dim=0).detach()
    alphas = [0.15, 0.20, 0.25, 0.30, 0.35]
    alphas = [a + 0.1 for a in alphas]
    percentiles = [0.9, 0.8, 0.7, 0.6, 0.5]


    n_plots = min(5, ys.shape[-1]) + 1

    fig, ax = plt.subplots(n_plots, figsize=convert_width((6, 6)), sharex=True)
    with torch.no_grad():
        for j in range(n_plots-1):
            ax[j].spines["bottom"].set_visible(False)
            ax[j].tick_params(axis="x", which="both", bottom=False)
            ax[j].plot(t_eval, median[:, j], **PRED_STYLE)
            ax[j].plot(data.train_t, data.train_r[:, j], **TRAIN_STYLE)
            ax[j].plot(data.valid_t, data.valid_r[:, j], **VALID_STYLE)
            ax[j].spines["top"].set_visible(False)
            ax[j].spines["right"].set_visible(False)
        for quant, alpha in zip(percentiles, alphas):
            lb = torch.quantile(ys, 1 - quant, dim=0)
            ub = torch.quantile(ys, quant, dim=0)
            for j in range(n_plots-1):
                lb_, ub_ = lb.T[j], ub.T[j]
                ax[j].fill_between(t_eval, lb_, ub_, alpha=alpha, color=FILL_COLOR)  # type: ignore

    ax[0].set_ylabel("$x(t)$")
    ax[1].set_ylabel("$y(t)$")
    ret_fig = (fig, ax)
    fig, ax = plt.subplots(figsize=convert_width((6, 6)), subplot_kw=dict(polar=True))
    ax.plot(*convert_to_polar(data.train_r), **TRAIN_STYLE)
    ax.plot(*convert_to_polar(data.valid_r), **VALID_STYLE)
    thetas, rs = convert_to_polar(ys)
    median_theta, _ = convert_to_polar(median)
    for quant, alpha in zip(percentiles, alphas):
        lb_r = torch.quantile(rs, 1 - quant, dim=0)
        ub_r = torch.quantile(rs, quant, dim=0)
        ax.fill_between(median_theta, lb_r, ub_r, alpha=alpha, color=FILL_COLOR)  # type: ignore

    ax.axis("off")
    ax.set_xlabel("$x(t)$")
    ax.set_ylabel("$y(t)$")
    fig.savefig(
        os.path.join(FIG_DIR, f"pred-plot{figsuffix}.pdf"),
        format="pdf",
        bbox_inches="tight",
    )

    return ret_fig

def wvf_from_samples(ys: Tensor, t_eval: Tensor) -> Tensor:
    x, y = ys[..., 0], ys[..., 1]
    Ixx = x**2 * ExtremeMassBBH().M
    Iyy = y**2 * ExtremeMassBBH().M
    Ixy = x * y * ExtremeMassBBH().M
    I = torch.stack([Ixx, Iyy, Ixy], dim=-1).detach().numpy()
    ddI = second_order_fd(np.transpose(I, (1, 0, 2)), t_eval[1] - t_eval[0])
    ddI = np.transpose(ddI, (1, 0, 2))
    ddIxx, ddIyy, _ = ddI[..., 0], ddI[..., 1], ddI[..., 2]
    wvf_pred = torch.as_tensor((ddIxx - ddIyy) * math.sqrt(4 * math.pi / 5))
    return wvf_pred

def view_waveform(ret_fig, soln_sample: tuple[Tensor, Tensor], data: Data, figsuffix: str):
    fig, ax= ret_fig
    sns.set(style="ticks")

    with torch.no_grad():
        ax.plot(data.train_t, data.train_y,'k.', markersize=0.5) #, **TRAIN_STYLE)

    ax.set_ylabel("$w(t)$")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return fig


def main():
    data = Data.load(CURR_DIR)
    nsamples = 1024
    nsde, epoch_iter = get_latest_model(nsamples)
    figsuffix = figname_suffix(epoch_iter)
    ckpt = os.path.join(CURR_DIR, "soln_sample.pkl")
    if not os.path.exists(ckpt):
        soln_sample = get_soln_samples(nsde, data, nsamples)
        with open(ckpt, "wb") as f:
            pkl.dump(soln_sample, f)
    else:
        with open(ckpt, "rb") as f:
            soln_sample = pkl.load(f)
    fig, ax = get_pred_plot(soln_sample, data, figsuffix)
    fig = view_waveform((fig, ax[2]), soln_sample, data, figsuffix)
    fig.savefig(
        os.path.join(FIG_DIR, f"waveform{figsuffix}.pdf"), format="pdf", bbox_inches="tight"
    )
    # not used in the final draft
    # plot_state_estimates(nsde, data, figsuffix)


if __name__ == "__main__":
    main()
