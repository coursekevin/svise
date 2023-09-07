from svise import odes, utils
import torch
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import os

save_dir = pathlib.Path(__file__).parent.resolve()
save_dir = os.path.join(save_dir, "data")

rs = 2021
torch.manual_seed(rs)
np.random.seed(rs)
torch.set_default_dtype(torch.float64)

d = 1024
test_ode = lambda t, x: odes.lorenz96(t, x)
tf = 10.0
x0 = np.random.normal(size=(d,))
noise = 0.5
degree = 5

system_choice = "Lorenz96"

num_data = 512
t = np.linspace(0.0, tf, num_data)
t_eval = torch.linspace(0.0, tf, 300)
dt = t[1] - t[0]
# generating data
test_ode(t, x0)
sol = solve_ivp(test_ode, (0.0, tf), x0, t_eval=t, atol=1e-9, rtol=1e-6)
y_true = torch.tensor(sol.y.T, dtype=torch.get_default_dtype())
sol_eval = solve_ivp(test_ode, (0.0, tf), x0, t_eval=t_eval, atol=1e-9, rtol=1e-6)
y_eval = torch.tensor(sol_eval.y.T, dtype=torch.get_default_dtype())
t = torch.tensor(t, dtype=torch.get_default_dtype())
W = torch.zeros(21)
noise_percent = 0.02
W[0] = 10.0
W[19] = -1.0
W[13] = 1.0
W[3] = -1.0

# simple measurement model
rank_list = 2 ** np.array([6, 7, 8, 9, 10])
# generated 0-4 then 5-9
num_indep = 5
for j in range(num_indep):
    for rank in rank_list:
        num_meas = d
        G = utils.make_random_matrix(d, rank)
        data_range = (
            (y_true @ G.t()).max(0).values - (y_true @ G.t()).min(0).values
        ) / 2
        var = (noise_percent * data_range) ** 2
        y_data = y_true @ G.t() + torch.randn_like(y_true) * torch.sqrt(var)
        save_dict = {
            "name": system_choice,
            "d": d,
            "tf": tf,
            "num_data": num_data,
            "t": t,
            "x0": x0,
            "y_true": y_true,
            "y_data": y_data,
            "var": var,
            "noise_percent": noise_percent,
            "G": G,
            "W": W,
            "degree": degree,
            "measurement_rank": rank,
        }
        system_name = system_choice.replace(" ", "_").lower()
        fname = f"{system_name}_rank{rank:04}_{j+5}.pt"
        torch.save(save_dict, os.path.join(save_dir, fname))
