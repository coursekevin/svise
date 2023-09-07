from svise import odes
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import time
import pathlib
import os
import torch
from scipy.special import comb

save_dir = pathlib.Path(__file__).parent.resolve()
save_dir = os.path.join(save_dir, "data")

# setting seeds
rs = 2021
np.random.seed(rs)
torch.manual_seed(rs)
torch.set_default_dtype(torch.float64)

# problem description

x0 = [0.0000, 1.0, 0.0, 0.0]
tf = 20
d = len(x0)
num_states = 4
d = num_states
test_ode = lambda t, x: odes.coupled_oscillator(t, x)
degree = 5
num_features = int(comb(num_states + degree, degree))
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

t_eval = np.linspace(0.0, tf, 4096)
sol_eval = solve_ivp(test_ode, (0.0, tf), x0, t_eval=t_eval, atol=1e-9, rtol=1e-6)
y_eval = torch.tensor(sol_eval.y.T, dtype=torch.get_default_dtype())
G = torch.eye(d)
data_range = ((y_eval @ G.t()).max(0).values - (y_eval @ G.t()).min(0).values) / 2
key = "Coupled linear"
num_data = 32
noise_percent = 0.1
t = np.linspace(0.0, tf, num_data)
dt = t[1] - t[0]
freq = (tf - 0.0) / num_data
# generating data
sol = solve_ivp(test_ode, (0.0, tf), x0, t_eval=t, atol=1e-9, rtol=1e-6)
y_true = torch.tensor(sol.y.T, dtype=torch.get_default_dtype())
# measurement function
var = (noise_percent * data_range) ** 2
# time stamps
t = torch.tensor(t, dtype=torch.get_default_dtype())
y_data = y_true @ G.t() + torch.randn_like(y_true) * torch.sqrt(var)
save_dict = {
    "name": key,
    "d": d,
    "tf": tf,
    "num_data": num_data,
    "t": t,
    "x0": x0,
    "hz": freq,
    "y_true": y_true,
    "y_data": y_data,
    "var": var,
    "noise_percent": noise_percent,
    "G": G,
    "W": W,
    "degree": degree,
}
system_name = key.replace(" ", "_").lower()
fname = f"{system_name}_{noise_percent*1000:03.0f}permille_{num_data:04}data.pt"
torch.save(save_dict, os.path.join(save_dir, fname))
