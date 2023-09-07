# dapper from: https://github.com/nansencenter/DAPPER
import argparse
import os
import pathlib
import time

import dapper.da_methods as da
import dapper.mods as modelling
from dapper.mods.integration import integrate_TLM
import numpy as np
import numpy.testing as npt
from scipy.integrate import solve_ivp
from svise import odes
import torch
from torch.autograd.functional import jacobian

save_dir = pathlib.Path(__file__).parent.resolve()

# random seed and dtype
rs = int(time.time())
torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description="DA Utility")
# path = os.path.join(save_dir, "data/hopf_bifurcation_010permille_0032data_00.pt")
default_dpath = os.path.join(
    save_dir, "data", "damped_cubic_oscillator_100permille_0128data_19.pt"
)
parser.add_argument("--dpath", type=str, help="data path", default=default_dpath)
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

# save name parameters
system_choice = data_dict["name"]
system_name = system_choice.replace(" ", "_").lower()
num_data = data_dict["num_data"]
noise_percent = data_dict["noise_percent"]
index = data_dict["index"]

fname = f"dapper_{system_name}_{noise_percent*1000:03.0f}permille_{num_data:04}data_{index:02}_{rs}rs"

# parameters for running experiment
d = data_dict["d"]
G = data_dict["G"]
Q_diag = data_dict["pnoise_cov"]
npt.assert_array_equal(G, torch.eye(d))
exp_ode = get_experiment(data_dict["name"])
np_ode = lambda x: np.stack(exp_ode(0.0, x), -1)
torch_ode = lambda x: torch.stack(exp_ode(0.0, x), -1)
yy = data_dict["y_data"].numpy()
xx = data_dict["y_true"].numpy()
Ny, num_states = yy.shape

print(88 * "-")
print(f"Running: {fname}")


def d2x_dtdx(x):
    """Jacobian of x (d-dim vector)"""
    xt = torch.tensor(x)
    jac = jacobian(torch_ode, xt)
    return jac.numpy()


def dstep_dx(x, t, dt):
    """Compute resolvent (propagator) of the TLM. I.e. the Jacobian of `step(x)`."""
    # return integrate_TLM(d2x_dtdx(x), dt, method="approx")
    return integrate_TLM(d2x_dtdx(x), dt, method="approx")  # forward euler


def corruption(t, x, dx, alpha: float, beta: float):
    dx_mod = dx
    dx_mod[..., 0] = dx[..., 0] - alpha * x[..., 1] + beta
    return dx_mod


# setting up tseq
tf = data_dict["tf"]
K = 10000
Ko = Ny - 1
dko = int(
    np.ceil(K / (Ko + 1))
)  # ratio of number of simulation points to observation points
dto = (data_dict["t"][1] - data_dict["t"][0]).item()
tseq = modelling.Chronology(Ko=Ko, dko=dko, dto=dto)
step = modelling.with_rk4(np_ode, autonom=True, stages=1)  # euler-maruyama instead
# I think this is used every time step based on HMM.tseq.dt (i.e. dynamic update rate)
# can check this by stepping through the extkf solution
Q_diag = data_dict["pnoise_cov"]
Q = modelling.CovMat(Q_diag.numpy(), kind="diag") 
# Q = modelling.CovMat(np.cov(xx.T)) * 1e-2
Dyn = {
    "M": num_states,
    "model": step,
    "linear": dstep_dx,
    "noise": modelling.GaussRV(C=Q),
}
# we want the initial condition to be almost useless
y_true = data_dict["y_true"]
data_range = (y_true.max(0).values - y_true.min(0).values) / 2
# chosen so that iEnKS does not sample such
var_init = 0.00 * data_range
P0 = modelling.CovMat(var_init, kind="diag")
# initial condition for hopf
if data_dict["name"] == "Hopf bifurcation":
    x0 = np.array(data_dict["x0"])
else:
    alpha = data_dict["alpha"]
    beta = data_dict["beta"]
    true_ode = lambda t, x: corruption(t, x, exp_ode(t, x), alpha, beta)
    sol_x0 = solve_ivp(
        exp_ode, (tseq.tto[0], tseq.tt[0]), data_dict["x0"], atol=1e-9, rtol=1e-6
    )
    x0 = sol_x0.y.T[-1]  # working backwrads to find the true initial condition
X0 = modelling.GaussRV(C=P0, mu=x0)
# setting up observation model
jj = np.arange(num_states)  # obs_inds
Obs = modelling.partial_Id_Obs(num_states, jj)
R = modelling.CovMat(data_dict["var"].numpy(), kind="diag")
Obs["noise"] = modelling.GaussRV(C=R)

# setting up HMM
HMM = modelling.HiddenMarkovModel(Dyn, Obs, tseq, X0)
xx, __ = HMM.simulate()
# creating list of da methods
# these are some sensible parameters, by no means optimal in all cases
infl = (data_range.numpy() * 0.01) ** 2
da_method = {
    "PartFilt": da.PartFilt(N=1000, reg=2.4, NER=0.3), # we only ended up using the PF
    "ExtKF": da.ExtKF(1.00),
}
da_data = {}
for name, method in da_method.items():
    print(f"Running: {name}")
    HMM_tmp = HMM.copy()
    if name == "iEnKS":
        # iEnKS doesn't work with set diffusion
        HMM_tmp.Dyn.noise.C = 0.0
    method.assimilate(HMM_tmp, xx, yy)
    try:  # check for smoothed soln
        mu = method.stats.mu.s  # smoothed
        std = method.stats.spread.s
        smoothed = True
    except AttributeError:  # else use recursive soln
        print(f"{name} doesn't have a smoothed solution")
        mu = method.stats.mu.a
        std = method.stats.spread.a
        smoothed = False
    if np.isnan(mu).any() or np.isnan(std).any():
        print("nans detected, suggest rerun.")
    da_data[f"{name}_mu"] = mu
    da_data[f"{name}_std"] = std
    da_data[f"{name}_t"] = tseq.tto - dto
    da_data[f"{name}_s"] = smoothed


save_name = os.path.join("results/models", fname)
torch.save(da_data, os.path.join(save_dir, save_name + ".pt"))
print(f"Done: {fname}")
print(88 * "-")
