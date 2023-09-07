from torch.serialization import save
from svise.sde_learning import *
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy.integrate import solve_ivp
import numpy as np
from svise import odes
import matplotlib.pyplot as plt
import time
import torch.optim.lr_scheduler as lr_scheduler
import pathlib
import argparse
import os
from tqdm import tqdm


torch.set_num_threads(12)
torch.set_default_dtype(torch.float64)


save_dir = pathlib.Path(__file__).parent.resolve()

# random seed and dtype
rs = int(time.time())
torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description="Newtonian ex. experiment util.")
parser.add_argument("--dpath", type=str, help="data path")
# set random seed, otherwise use input
parser.add_argument("--rs", type=int, help="random seed", default=rs)
args = parser.parse_args()
rs = args.rs

torch.manual_seed(rs)
np.random.seed(rs)

# dpath = os.path.join(save_dir, "data/coupled_linear_100permille_0032data.pt")

dpath = args.dpath
data_dict = torch.load(dpath)

system_choice = data_dict["name"]
d = data_dict["d"]
num_data = data_dict["num_data"]
t = data_dict["t"]
G = data_dict["G"]
y_data = data_dict["y_data"][..., :2]
var = data_dict["var"]
noise_percent = data_dict["noise_percent"]
x0 = data_dict["x0"]
degree = data_dict["degree"]
t0 = float(t.min())
tf = float(t.max())

t_eval = torch.linspace(t0, tf, 300)
sol_eval = solve_ivp(
    odes.coupled_oscillator, (0.0, tf), x0, t_eval=t_eval, atol=1e-9, rtol=1e-6
)
y_eval = torch.tensor(sol_eval.y.T, dtype=torch.get_default_dtype())

G = torch.eye(4)[:2]

n_reparam_samples = 32
input_labels = ["x_1", "x_2", "v_1", "v_2"]
sde = SparsePolynomialIntegratorSDE(
    d,
    (t.min(), t.max()),
    degree=5,
    n_reparam_samples=n_reparam_samples,
    G=G,
    num_meas=G.shape[0],
    measurement_noise=var[:2],
    input_labels=input_labels,
    n_tau=200,
    train_t=t,
    train_x=y_data,
)
sde.train()
num_iters = 20000
transition_iters = 5000
# assert transition_iters < num_iters
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
for epoch in tqdm(range(num_epochs)):
    for t_batch, y_batch in train_loader:
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
            print_loss = True
        else:
            print_loss = False
        loss = -sde.elbo(t_batch, y_batch, beta, num_data, print_loss=print_loss)
        loss.backward()
        optimizer.step()
fig, axs = plt.subplots(2, 1)
sde.eval()
mu = sde.marginal_sde.mean(t_eval)
covar = sde.marginal_sde.K(t_eval)
var_state = covar.diagonal(dim1=-2, dim2=-1)
print(
    f"iter: {j:05}/{num_iters:05} | loss: {loss.item():04.2f} | mode: {train_mode} | time: {time.time() - start_time:.2f} | beta: {beta:.2f} | lr: {scheduler.get_last_lr()[0]:.5f}",
    flush=True,
)
sde.sde_prior.update_sparse_index()
print_str = f"Learnt governing equations: {sde.sde_prior.feature_names}"
print(print_str)
sde.sde_prior.reset_sparse_index()
start_time = time.time()
lb = mu - 2 * var_state.sqrt()
ub = mu + 2 * var_state.sqrt()
with torch.no_grad():
    [ax.cla() for ax in axs]
    axs[0].plot(t, y_data, "C1o", alpha=0.5)
    axs[0].plot(t_eval, mu, "C0-")
    axs[0].plot(t_eval, y_eval, "k--")
    sde_drift = sde.sde_prior(t_eval, mu)
    for j in range(d):
        axs[0].fill_between(t_eval, lb[:, j], y2=ub[:, j], color="C0", alpha=0.2)
        axs[1].plot(t_eval, sde_drift[..., j].t(), f"C{j}-")
    axs[1].plot(t_eval, odes.torch_ode(t_eval, y_eval, odes.coupled_oscillator), "k--")

system_name = system_choice.replace(" ", "_").lower()
fname = (
    f"svise_{system_name}_{noise_percent*1000:03.0f}permille_{num_data:04}data_{rs}rs"
)
save_name = os.path.join("results/pngs", fname)
plt.savefig(os.path.join(save_dir, save_name + ".png"))
save_name = os.path.join("results/models", fname)
torch.save(sde.state_dict(), os.path.join(save_dir, save_name + ".pt"))
