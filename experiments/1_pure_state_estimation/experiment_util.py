# this script takes in a data file path and trains an svise model on it
import argparse
import os
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from svise import odes
from svise.sde_learning import *

save_dir = pathlib.Path(__file__).parent.resolve()

# random seed and dtype
rs = int(time.time())
torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description="Pure state estimation SVISE run script")
parser.add_argument("--dpath", type=str, help="data path")
# set random seed, otherwise use input
parser.add_argument("--rs", type=int, help="random seed", default=rs)
args = parser.parse_args()
rs = args.rs
torch.manual_seed(rs)
np.random.seed(rs)


def get_experiment(system_name):
    if system_name == "Lorenz63":
        params = (10, 8 / 3, 28)
        test_ode = lambda t, x: odes.lorenz63(t, x, *params)
    elif system_name == "Damped linear oscillator":
        test_ode = lambda t, x: odes.damped_oscillator(t, x, osc_type="linear")
    elif system_name == "Damped cubic oscillator":
        test_ode = lambda t, x: odes.damped_oscillator(t, x, osc_type="cubic")
    elif system_name == "Hopf bifurcation":
        test_ode = lambda t, x: odes.hopf_ode(t, x)
    elif system_name == "Selkov glycolysis model":
        test_ode = lambda t, x: odes.selkov(t, x)
    elif system_name == "Duffing oscillator":
        test_ode = lambda t, x: odes.duffing(t, x)
    elif system_name == "Coupled linear":
        test_ode = lambda t, x: odes.coupled_oscillator(t, x)
    else:
        raise ValueError(f"Unknown system name: {system_name}")
    return test_ode


data_dict = torch.load(args.dpath)

system_choice = data_dict["name"]
d = data_dict["d"]
num_data = data_dict["num_data"]
t = data_dict["t"]
G = data_dict["G"]
y_data = data_dict["y_data"]
var = data_dict["var"]
noise_percent = data_dict["noise_percent"]
x0 = data_dict["x0"]
index = data_dict["index"]
degree = data_dict["degree"]
Q_diag = data_dict["pnoise_cov"]
t0 = float(t.min())
tf = float(t.max())

t_eval = torch.linspace(t0, tf, 300)

print_str = f"Running '{system_choice}' experiment with {num_data} data points and {var} Gaussian noise. Random seed {rs}."
print(len(print_str) * "-")
print(print_str, flush=True)
drift_function = lambda t, x: odes.torch_ode(t, x, get_experiment(system_choice))
print(f"System choice: {system_choice}")
# simple measurement model
n_reparam_samples = 32
sde = StateEstimator(
    d,
    (t0, tf),
    n_reparam_samples=n_reparam_samples,
    G=G,
    drift=drift_function,
    num_meas=G.shape[0],
    measurement_noise=var,
    train_t=t,
    train_x=y_data,
    Q_diag=Q_diag,
)
if system_choice == "Damped cubic oscillator":
    # need better eval initialization for this problem
    nn.init.constant_(sde.marginal_sde.eigenvals.b, -2.2522 * 2)
sde.train()
num_iters = 20000
transition_iters = 5000
assert transition_iters < num_iters
num_mc_samples = int(min(128, num_data))
summary_freq = 1000
scheduler_freq = transition_iters // 2
lr = 1e-3
optimizer = torch.optim.Adam(
    [
        {"params": sde.state_params()},
        {"params": sde.sde_params(), "lr": 1e-2},
    ],
    lr=lr,
)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
start_time = time.time()
train_dataset = TensorDataset(t, y_data)
train_loader = DataLoader(train_dataset, batch_size=num_mc_samples, shuffle=True)
num_epochs = num_iters // len(train_loader)
j = 0

fig, ax = plt.subplots()


def print_status(sde, show_plot):
    ax.cla()
    with torch.no_grad():
        mu = sde.marginal_sde.mean(t_eval)
        covar = sde.marginal_sde.K(t_eval)
        var = covar.diagonal(dim1=-2, dim2=-1)
        lb = mu - 2 * var.sqrt()
        ub = mu + 2 * var.sqrt()
        ax.plot(t, y_data, "C1o", alpha=0.5)
        ax.plot(t_eval, mu, "C0-")
        sde_drift = sde.sde_prior(t_eval, mu)
        for j in range(d):
            ax.fill_between(t_eval, lb[:, j], y2=ub[:, j], color="C0", alpha=0.2)
        if show_plot:
            plt.pause(1 / 60)


with tqdm(total=num_iters) as pbar:
    for epoch in range(num_epochs):
        for t_batch, y_batch in train_loader:
            j += 1
            optimizer.zero_grad()
            idx = np.random.choice(np.arange(num_data), num_mc_samples, replace=False)
            if j < (transition_iters):
                # beta warmup iters
                beta = min(1.0, (1.0 * j) / (transition_iters))
                train_mode = "beta"
            else:
                beta = 1.0
                train_mode = "full"
            if j % scheduler_freq == 0:
                scheduler.step()
            if j % summary_freq == 0 or j == 1:
                print_loss = True
                sde.eval()
                print_status(sde, show_plot=False) # this was set to True when creating original models
                sde.train()
            else:
                print_loss = False
            loss = -sde.elbo(t_batch, y_batch, beta, num_data, print_loss=print_loss)
            loss.backward()
            optimizer.step()
            pbar.update(1)

sde.eval()
print_status(sde, show_plot=True)
print(
    f"iter: {j:05}/{num_iters:05} | loss: {loss.item():04.2f} | mode: {train_mode} | time: {time.time() - start_time:.2f} | beta: {beta:.2f} | lr: {scheduler.get_last_lr()[0]:.5f}",
    flush=True,
)
sde.train()  # back in train mode

system_name = system_choice.replace(" ", "_").lower()
fname = f"svise_{system_name}_{noise_percent*1000:03.0f}permille_{num_data:04}data_{index:02}_{rs}rs"
save_name = os.path.join("results/pngs", fname)
plt.savefig(os.path.join(save_dir, save_name + ".png"))
save_name = os.path.join("results/models", fname)
torch.save(sde.state_dict(), os.path.join(save_dir, save_name + ".pt"))
print("Done.")
