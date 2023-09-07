import os
import pathlib

import numpy as np
import numpy.random as npr
import torch
from scipy.integrate import solve_ivp
from scipy.special import comb
from scipy.interpolate import interp1d
from tqdm import tqdm

from svise import odes
from svise import utils

save_dir = pathlib.Path(__file__).parent.resolve()
save_dir = os.path.join(save_dir, "data")

# setting seeds
rs = 2023
np.random.seed(rs)
torch.manual_seed(rs)
torch.set_default_dtype(torch.float64)

degree = 5


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


system_list = [
    "Damped linear oscillator",
    "Damped cubic oscillator",
    "Lorenz63",
    "Hopf bifurcation",
    "Selkov glycolysis model",
    "Duffing oscillator",
    "Coupled linear",
]

num_indep_datasets = 20
for key in tqdm(system_list):
    for i in range(num_indep_datasets):
        num_data = 128
        # tuning for max corruption
        d, test_ode, tf, x0, _, W = get_experiment(key)
        num_features = comb(d + degree, degree)
        t = np.linspace(0.0, tf, num_data)
        dt = t[1] - t[0]
        freq = (tf - 0.0) / num_data
        # generating data
        sol = solve_ivp(test_ode, (0.0, tf), x0, t_eval=t, atol=1e-9, rtol=1e-6)
        y_sample_trajectory = torch.tensor(sol.y.T, dtype=torch.get_default_dtype())
        # measurement function
        G = torch.eye(d)
        noise_percent = 0.10
        process_noise_percent = 0.01
        data_range = (
            (y_sample_trajectory @ G.t()).max(0).values
            - (y_sample_trajectory @ G.t()).min(0).values
        ) / 2
        var = (noise_percent * data_range) ** 2
        # time stamps
        t = torch.tensor(t, dtype=torch.get_default_dtype())
        L = torch.eye(d)
        Q = (process_noise_percent * data_range) ** 2
        em_sol = utils.EulerMaruyama(
            f=lambda t, x: odes.torch_ode(t, x, test_ode),
            L=torch.eye(d),
            Q=torch.diag(process_noise_percent * data_range) ** 2,
            t_span=(0.0, tf),
            x0=torch.tensor(x0).unsqueeze(0),
            dt=tf / (10000 - 1),
        )

        def interpolate(t_sol, y_sol, t_eval):
            f_eval = torch.stack(
                [
                    torch.tensor(interp1d(t_sol, y_sol[:, i])(t_eval))
                    for i in range(y_sol.shape[-1])
                ],
                dim=-1,
            )
            return f_eval

        em_sol.t[-1] += 1e-6
        em_sol.t[0] -= 1e-6
        y_true = interpolate(em_sol.t, em_sol.x[0], t)
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
            "pnoise_cov": Q,
            "G": G,
            "W": W,
            "degree": degree,
            "index": i,
        }
        system_name = key.replace(" ", "_").lower()
        # end_string = "alpha" + str(alpha_bnds).replace(".","-") + "beta" + str(beta_bnds).replace(".","-")
        fname = f"{system_name}_{noise_percent * 1000:03.0f}permille_{num_data:04}data_{i:02}.pt"
        torch.save(save_dict, os.path.join(save_dir, fname))
    #     plt.plot(t, y_data, "bo", alpha=0.05)
    #     plt.plot(t, y_true, "b-", alpha=0.05)
    #     plt.plot(t, y_true, "k-")
    # plt.show()
    # break
