# this script takes in a data file path and trains an sde model on it
from svise.sde_learning import *
from svise import odes
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pathlib
import argparse

from scipy.special import comb

save_dir = pathlib.Path(__file__).parent.resolve()

# random seed and dtype
rs = int(time.time())
torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description="Linear oscillator experiment utility")
parser.add_argument("--dpath", type=str, help="data path")
# set random seed, otherwise use input
parser.add_argument("--rs", type=int, help="random seed", default=rs)
args = parser.parse_args()
rs = args.rs
torch.manual_seed(rs)
np.random.seed(rs)

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
t0 = float(t.min())
tf = float(t.max())


def get_experiment(system_name):
    if system_name == "Lorenz63":
        num_states = 3
        params = (10, 8 / 3, 28)
        test_ode = lambda t, x: odes.lorenz63(t, x, *params)
        tf = 10.0
        x0 = [-8.0, 7.0, 27.0]
        noise = 1e-1
        num_features = int(comb(num_states + degree, degree))
        W = torch.zeros(num_states, num_features)
        W[0, 1] = -10.0
        W[0, 2] = 10.0
        W[1, 1] = 28.0
        W[1, 6] = -1.0
        W[1, 2] = -1.0
        W[2, 5] = 1.0
        W[2, 3] = -8 / 3
    elif system_name == "Damped linear oscillator":
        num_states = 2
        test_ode = lambda t, x: odes.damped_oscillator(t, x, osc_type="linear")
        tf = 20.0
        x0 = [2.5, -5.0]  # these intial conditions make for better eq learning
        noise = 1e-1
        num_features = int(comb(num_states + degree, degree))
        W = torch.zeros(num_states, num_features)
        W[0, 1] = -0.1
        W[0, 2] = 2.0
        W[1, 1] = -2.0
        W[1, 2] = -0.1
    elif system_name == "Damped cubic oscillator":
        num_states = 2
        test_ode = lambda t, x: odes.damped_oscillator(t, x, osc_type="cubic")
        tf = 25.0
        x0 = [0.0, -1.0]  # these intial conditions make for better eq learning
        noise = 1e-2
        num_features = int(comb(num_states + degree, degree))
        W = torch.zeros(num_states, num_features)
        W[0, 6] = -0.1
        W[0, 9] = 2.0
        W[1, 6] = -2.0
        W[1, 9] = -0.1
    elif system_name == "Hopf bifurcation":
        num_states = 2
        test_ode = lambda t, x: odes.hopf_ode(t, x)
        tf = 20.0
        x0 = [2.0, 2.0]
        noise = 1e-1
        num_features = int(comb(num_states + degree, degree))
        W = torch.zeros(num_states, num_features)
        W[0, 1] = 0.5
        W[0, 2] = 1.0
        W[0, 6] = -1.0
        W[0, 8] = -1.0
        W[1, 1] = -1.0
        W[1, 2] = 0.5
        W[1, 7] = -1.0
        W[1, 9] = -1.0
    elif system_name == "Selkov glycolysis model":
        num_states = 2
        test_ode = lambda t, x: odes.selkov(t, x)
        tf = 30.0
        x0 = [0.7, 1.25]
        noise = 1e-2
        num_features = int(comb(num_states + degree, degree))
        W = torch.zeros(num_states, num_features)
        W[0, 1] = -1.0
        W[0, 2] = 0.08
        W[0, 7] = 1.0
        W[1, 0] = 0.6
        W[1, 2] = -0.08
        W[1, 7] = -1.0
    elif system_name == "Duffing oscillator":
        num_states = 2
        test_ode = lambda t, x: odes.duffing(t, x)
        tf = 20.0
        x0 = [3.0, 2.0]
        noise = 0.5
        num_features = int(comb(num_states + degree, degree))
        W = torch.zeros(num_states, num_features)
        W[0, 2] = 1.0
        W[1, 1] = -1.0
        W[1, 6] = -1.0
        W[1, 2] = -0.35
    elif system_name == "Coupled linear":
        num_states = 4
        test_ode = lambda t, x: odes.coupled_oscillator(t, x)
        tf = 20.0
        x0 = [0.0000, 1.0, 0.0, 0.0]
        noise = 1e-1
        k1 = 4
        k2 = 2
        k3 = 4
        m1 = 1
        m2 = 1.0
        num_features = int(comb(num_states + degree, degree))
        W = torch.zeros(num_states, num_features)
        W[0, 3] = 1.0
        W[1, 4] = 1.0
        W[2, 1] = -(k1 + k2) / m1
        W[2, 2] = k2 / m1
        W[3, 1] = k2 / m2
        W[3, 2] = -(k2 + k3) / m2
    else:
        raise ValueError(f"Unknown system name: {system_name}")
    return num_states, test_ode, tf, x0, noise, W


d, test_ode, tf, x0, _, W = get_experiment(system_choice)
t_eval = torch.linspace(t0, tf, 300)
sol_eval = solve_ivp(test_ode, (0.0, tf), x0, t_eval=t_eval, atol=1e-9, rtol=1e-6)
y_eval = torch.tensor(sol_eval.y.T, dtype=torch.get_default_dtype())

print_str = f"Running '{system_choice}' experiment with {num_data} data points and {var} Gaussian noise. Random seed {rs}."
print(len(print_str) * "-")
print(print_str, flush=True)
# simple measurement model
n_reparam_samples = 32
sde = SparsePolynomialSDE(
    d,
    (t0, tf),
    degree=5,
    n_reparam_samples=n_reparam_samples,
    G=G,
    num_meas=G.shape[0],
    measurement_noise=var,
    train_t=t,
    train_x=y_data,
)
sde.train()
num_iters = 20000
transition_iters = 5000
assert transition_iters < num_iters
num_mc_samples = int(min(128, num_data))
summary_freq = 1000
scheduler_freq = transition_iters // 2
lr = 1e-3
optimizer = torch.optim.Adam(
    [{"params": sde.state_params()}, {"params": sde.sde_params(), "lr": 1e-2},], lr=lr
)
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
start_time = time.time()
train_dataset = TensorDataset(t, y_data)
train_loader = DataLoader(train_dataset, batch_size=num_mc_samples, shuffle=True)
num_epochs = num_iters // len(train_loader)
j = 0
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
        if j % summary_freq == 0:
            print_loss = True
        else:
            print_loss = False
        loss = -sde.elbo(t[idx], y_data[idx], beta, num_data, print_loss=print_loss)
        loss.backward()
        optimizer.step()
        if j % summary_freq == 0:
            sde.eval()
            mu = sde.marginal_sde.mean(t_eval)
            covar = sde.marginal_sde.K(t_eval)
            var = covar.diagonal(dim1=-2, dim2=-1)
            print(
                f"iter: {j:05}/{num_iters:05} | loss: {loss.item():04.2f} | mode: {train_mode} | time: {time.time() - start_time:.2f} | beta: {beta:.2f} | lr: {scheduler.get_last_lr()[0]:.5f}",
                flush=True,
            )
            sde.sde_prior.update_sparse_index()
            print_str = f"Learnt governing equations: {sde.sde_prior.feature_names}"
            print(print_str)
            sde.sde_prior.reset_sparse_index()
            start_time = time.time()
            lb = mu - 2 * var.sqrt()
            ub = mu + 2 * var.sqrt()
            sde.train()  # back in train mode
fig, axs = plt.subplots(2, 1)
sde.eval()
with torch.no_grad():
    [ax.cla() for ax in axs]
    axs[0].plot(t, y_data, "C1o", alpha=0.5)
    axs[0].plot(t_eval, mu, "C0-")
    axs[0].plot(t_eval, y_eval, "k--")
    sde_drift = sde.sde_prior(t_eval, mu)
    for j in range(d):
        axs[0].fill_between(t_eval, lb[:, j], y2=ub[:, j], color="C0", alpha=0.2)
        axs[1].plot(t_eval, sde_drift[..., j].t(), f"C{j}-")
    axs[1].plot(t_eval, odes.torch_ode(t_eval, y_eval, test_ode), "k--")
sde.sde_prior.update_sparse_index()
print_str = f"Learnt governing equations: {sde.sde_prior.feature_names}"
print(print_str)
print(len(print_str) * "=")

system_name = system_choice.replace(" ", "_").lower()
fname = f"svise_{system_name}_{noise_percent*1000:03.0f}permille_{num_data:04}data_{index:02}_{rs}rs"
save_name = os.path.join("results/pngs", fname)
plt.savefig(os.path.join(save_dir, save_name + ".png"))
save_name = os.path.join("results/models", fname)
torch.save(sde.state_dict(), os.path.join(save_dir, save_name + ".pt"))
