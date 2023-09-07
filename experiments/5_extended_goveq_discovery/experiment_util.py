# this script takes in a data file path and trains an sde model on it
from svise.sde_learning import *
from svise.variationalsparsebayes.sparse_glm import SparsePolynomialNeighbour1D
from svise import quadrature
from svise import odes
from svise.utils import solve_least_squares
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import time
from time import perf_counter as pfc
import os
import pathlib
import argparse

save_dir = pathlib.Path(__file__).parent.resolve()

# random seed and dtype
rs = int(time.time())
torch.set_default_dtype(torch.float64)
parser = argparse.ArgumentParser(description="High-dim experiment utility")
parser.add_argument("--dpath", type=str, nargs="+", help="data path")
# set random seed, otherwise use input
parser.add_argument("--rs", type=int, help="random seed", default=rs)
args = parser.parse_args()
rs = args.rs
device = torch.device("cuda")
dpath_list = args.dpath
for ind, dpath in enumerate(dpath_list):
    torch.manual_seed(rs+ind)
    np.random.seed(rs+ind)
    exp_num = int(dpath.split("_")[-1][:-3])
    # loading data dictionary
    data_dict = torch.load(dpath)
    system_choice = data_dict["name"]
    d = data_dict["d"]
    num_data = data_dict["num_data"]
    t = data_dict["t"]
    G = data_dict["G"]
    y_data = data_dict["y_data"]
    var = data_dict["var"]
    noise_percent = data_dict["noise_percent"]
    degree = data_dict["degree"]
    degree = 2
    x0 = data_dict["x0"]
    rank = data_dict["measurement_rank"]
    t0 = float(t.min())
    tf = float(t.max())
    print_str = f"Running '{system_choice}' experiment with rank {rank} data points on {device}. Random seed {rs+ind}."
    print(len(print_str) * "-")
    print(print_str, flush=True)
    # simple measurement model
    n_reparam_samples = 32
    Q_diag = torch.ones(d) * 1e-0
    diffusion_prior = SparseDiagonalDiffusionPrior(d, Q_diag, n_reparam_samples, 1e-5)
    # getting a good guess for the initial state
    b = G.t().mul(var) @ y_data.t()
    A = G.t().mul(var) @ G + torch.eye(d) * 1e-2
    C = torch.linalg.cholesky(A)
    z_data = torch.cholesky_solve(b, C).t()
    marginal_sde = DiagonalMarginalSDE(
        d,
        (t0, tf),
        diffusion_prior=diffusion_prior,
        model_form="GLM",
        n_tau=200,
        learn_inducing_locations=False,
        train_x=t,
        train_y=z_data,
    )
    num_meas = G.shape[0]
    quad_scheme = quadrature.UnbiasedGaussLegendreQuad(t0, tf, 128, quad_percent=0.8)
    likelihood = IndepGaussLikelihood(G, num_meas, var)
    features = SparsePolynomialNeighbour1D(d, degree=degree, input_labels=["x"])
    m, dmdt = marginal_sde.mean(t, return_grad=True)
    sde_prior = SparseNeighbourGLM(
        d,
        SparseFeatures=features,
        n_reparam_samples=n_reparam_samples,
        tau=1e-5,
        train_x=m,
        train_y=dmdt,
    )
    sde = SDELearner(
        marginal_sde=marginal_sde,
        likelihood=likelihood,
        quad_scheme=quad_scheme,
        sde_prior=sde_prior,
        diffusion_prior=diffusion_prior,
        n_reparam_samples=n_reparam_samples,
    )
    sde.train()
    sde.to(device)
    sde.sde_prior.reset_sparse_index()
    num_iters = 5000
    transition_iters = 1250
    # assert transition_iters < num_iters
    num_mc_samples = int(min(128, num_data))
    summary_freq = 100
    scheduler_freq = transition_iters // 2
    lr = 1e-2
    optimizer = torch.optim.Adam(
        [{"params": sde.state_params()}, {"params": sde.sde_params(), "lr": 1e-1},],
        lr=lr,
    )
    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    train_dataset = TensorDataset(t, y_data)
    train_loader = DataLoader(
        train_dataset, batch_size=num_mc_samples, shuffle=True, num_workers=0
    )
    num_epochs = num_iters // len(train_loader)
    j = 0
    # todo: try implementing a manaul gradient for the polynomial dictionary
    start_time = pfc()
    for epoch in range(num_epochs):
        for t_batch, y_batch in train_loader:
            t_batch, y_batch = t_batch.to(device), y_batch.to(device)
            j += 1
            optimizer.zero_grad()
            if j < (transition_iters):
                # beta warmup iters
                beta = min(1.0, (1.0 * j) / (transition_iters))
                train_mode = "beta"
            else:
                beta = 1.0
                train_mode = "full"
            if j % scheduler_freq == 0:
                scheduler.step()
            if j % summary_freq == 0:
                print_loss = False
            else:
                print_loss = False
            loss = -sde.elbo(t_batch, y_batch, beta, num_data, print_loss=print_loss)
            if j % summary_freq == 0:
                total_time = pfc() - start_time
                print(
                    f"iter: {j:05} | loss: {loss.item():.2f} | time: {total_time:.1f}",
                    flush=True,
                )
                sde.eval()
                sde.to("cpu")
                sde.sde_prior.resample_weights()
                sde.sde_prior.update_sparse_index()
                print_str = f"Learnt governing equations: {sde.sde_prior.feature_names}"
                print(print_str)
                sde.train()
                sde.to(device)
                sde.sde_prior.reset_sparse_index()
                print(len(print_str) * "=")
                start_time = pfc()
            loss.backward()
            optimizer.step()
    sde.eval()
    sde.to("cpu")
    sde.sde_prior.resample_weights()
    sde.sde_prior.update_sparse_index()
    print_str = f"Learnt governing equations: {sde.sde_prior.feature_names}"
    print(print_str)
    print(len(print_str) * "=")
    system_name = system_choice.replace(" ", "_").lower()
    fname = f"svise_{system_name}_rank{rank}_{exp_num:02}_{rs+ind}rs"
    save_name = os.path.join("results", fname)
    torch.save(sde.state_dict(), os.path.join(save_dir, save_name + ".pt"))
    print(f"Done, saved to: {save_name}")
