from svise import odes
from svise.sde_learning import *
import os
import pathlib
import torch
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import matplotlib 
import matplotlib.pyplot as plt
from tqdm import tqdm

from svise import odes
from svise import utils

import seaborn as sns

torch.set_default_dtype(torch.float64)

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

def convert_width(fsize: tuple[float, float], page_scale: float = 0.5) -> tuple[float, float]:
    rescale_width = 2 * WIDTH_INCHES * page_scale
    width = fsize[0] 
    return tuple(size * rescale_width / width for size in fsize)

degree = 5


def interpolate(t_sol, y_sol, t_eval):
    f_eval = torch.stack(
        [
            torch.tensor(interp1d(t_sol, y_sol[:, i])(t_eval))
            for i in range(y_sol.shape[-1])
        ],
        dim=-1,
    )
    return f_eval


def corruption(t, x, dx, alpha: float, beta: float):
    dx_mod = dx
    dx_mod[..., 0] = dx[..., 0] - alpha * x[..., 1] + beta
    return dx_mod


corruption_tuning = {
    "Damped linear oscillator": (0.5, 0.5),  # .6
    "Damped cubic oscillator": (0.3, 0.3),
    "Lorenz63": (5.0, 5.0),  # 6.0
    "Hopf bifurcation": (0.25, 0.25),
    "Selkov glycolysis model": (0.08, 0.08),
    "Duffing oscillator": (0.5, 0.5),  # not bad
    "Coupled linear": (0.5, 0.5),
}


def get_experiment(system_name):
    if system_name == "Lorenz63":
        num_states = 3
        params = (10, 8 / 3, 28)
        test_ode = lambda t, x: odes.lorenz63(t, x, *params)
        tf = 10.0
        x0 = [-8.0, 7.0, 27.0]
    elif system_name == "Damped linear oscillator":
        num_states = 2
        test_ode = lambda t, x: odes.damped_oscillator(t, x, osc_type="linear")
        tf = 20.0
        x0 = [2.5, -5.0]  # these intial conditions make for better eq learning
    elif system_name == "Damped cubic oscillator":
        num_states = 2
        test_ode = lambda t, x: odes.damped_oscillator(t, x, osc_type="cubic")
        tf = 25.0
        x0 = [0.0, -1.0]  # these intial conditions make for better eq learning
    elif system_name == "Hopf bifurcation":
        num_states = 2
        test_ode = lambda t, x: odes.hopf_ode(t, x)
        tf = 20.0
        x0 = [2.0, 2.0]
    elif system_name == "Selkov glycolysis model":
        num_states = 2
        test_ode = lambda t, x: odes.selkov(t, x)
        tf = 30.0
        x0 = [0.7, 1.25]
    elif system_name == "Duffing oscillator":
        num_states = 2
        test_ode = lambda t, x: odes.duffing(t, x)
        tf = 20.0
        x0 = [3.0, 2.0]
    elif system_name == "Coupled linear":
        num_states = 4
        test_ode = lambda t, x: odes.coupled_oscillator(t, x)
        tf = 20.0
        x0 = [0.0000, 1.0, 0.0, 0.0]
    else:
        raise ValueError(f"Unknown system name: {system_name}")
    # corrupted_ode = lambda t, x: corruption(t, x, test_ode(t, x))
    return num_states, test_ode, tf, x0


# get list of data files
current_dir = pathlib.Path(__file__).parent.resolve()
data_dir = "data"
save_dir = os.path.join(current_dir, "data")
data_paths = []
for path, _, files in os.walk(os.path.join(current_dir, data_dir)):
    for f in files:
        if ".pt" in f:
            data_paths.append(os.path.join(current_dir, data_dir, f))
post_process_dir = os.path.join(current_dir, "results", "post_processing")

system_list = [
    "Damped linear oscillator",  # no need linear forward model
    "Damped cubic oscillator",
    "Lorenz63",
    "Hopf bifurcation",  # difficult to get initial condition correct for other methods
    "Selkov glycolysis model",
    "Duffing oscillator",
    "Coupled linear",  # no need for linear forward model
]

num_indep_datasets = 20
num_data = 128
noise_percent = 0.10
process_noise_percent = 0.01
num_samples = 1000
for name in tqdm(system_list):
    d, test_ode, tf, x0 = get_experiment(name)
    tf *= 0.25
    t_eval = np.linspace(0, tf, 1000)
    sol = solve_ivp(test_ode, (0.0, tf), x0, t_eval=t_eval, atol=1e-9, rtol=1e-6)
    y_sample_trajectory = torch.tensor(sol.y.T, dtype=torch.get_default_dtype())
    # measurement function
    G = torch.eye(d)
    # high noise
    noise_percent = 0.10
    process_noise_percent = 0.01
    data_range = (
        (y_sample_trajectory @ G.t()).max(0).values
        - (y_sample_trajectory @ G.t()).min(0).values
    ) / 2
    # --------------------------------------------------------------------------------
    # uncorrupted trajectories
    # --------------------------------------------------------------------------------
    em_uc = utils.EulerMaruyama(
        f=lambda t, x: odes.torch_ode(t, x, test_ode),
        L=torch.eye(d),
        Q=torch.diag(process_noise_percent * data_range) ** 2,
        t_span=(0.0, tf),
        x0=torch.tensor(x0).repeat(num_samples, 1),
        dt=tf / (10000 - 1),
    )

    # --------------------------------------------------------------------------------
    # corrupted trajectories
    # --------------------------------------------------------------------------------

    # sampling from corruption definition
    alpha_bnds, beta_bnds = corruption_tuning[name]
    alpha = torch.rand(num_samples) * alpha_bnds - alpha_bnds / 2
    beta = torch.rand(num_samples) * beta_bnds - beta_bnds / 2

    def corrupted_dynamics(t, x):
        return corruption(t, x, odes.torch_ode(t, x, test_ode), alpha, beta)

    corrupted_dynamics(0.0, torch.tensor(x0).repeat(num_samples, 1))

    em_cr = utils.EulerMaruyama(
        f=corrupted_dynamics,
        L=torch.eye(d),
        Q=torch.diag(process_noise_percent * data_range) ** 2,
        t_span=(0.0, tf),
        x0=torch.tensor(x0).repeat(num_samples, 1),
        dt=tf / (10000 - 1),
    )

    c1_color = "royalblue"
    c0_color = "gold"
    c1_rgb = matplotlib.colors.to_rgb(c1_color)
    c0_rgb = matplotlib.colors.to_rgb(c0_color)
    fig, ax = plt.subplots(1, 1, figsize=convert_width((8, 6), page_scale=0.25))
    plot_design = (
        {
            "color": "C1",
            "hatch": "",
            "facecolor": (*c1_rgb, 0.8),
            "edgecolor": (*matplotlib.colors.to_rgb(c1_color), 1.0),
        },
        {
            "color": "C0",
            "facecolor": (*c0_rgb, 0.8),
            "edgecolor": (*matplotlib.colors.to_rgb(c0_color), 1.0),
        },
    )

    for em, em_type, psettings in zip(
        (em_cr, em_uc), ("Corrupted", "Uncorrupted"), plot_design
    ):
        for i in range(d):
            if i == 0:
                label = f"{em_type} dynamics"
            else:
                label = None
            lb = torch.quantile(em.x[..., i], 0.1, dim=0)
            ub = torch.quantile(em.x[..., i], 0.9, dim=0)
            # plt.fill_between(em.t, lb, ub, facecolor=(1,1,1,1.0), edgecolor=(1,1,1,0.0)) # getting white background
            plt.fill_between(em.t, lb, ub, **psettings, linewidth=1.0, label=label)
    # for em, em_type, pcolor in zip(
    #     (em_cr, em_uc), ("Corrupted", "Uncorrupted"), ("C1", "C0")
    # ):
    #     for i in range(d):
    #         md = torch.quantile(em.x[..., i], 0.5, dim=0)
    #         if i == 0:
    #             plt.plot(em.t, md, "-", label=f"{em_type} dynamics", color=pcolor)
    #         else:
    #             plt.plot(em.t, md, "-", color=pcolor)
    ax.set_xlabel("$t$ (sec)")
    ax.set_frame_on(False)
    # ax.axis("off")
    ax.set_yticks([])
    xmin, xmax = ax.get_xaxis().get_view_interval()
    ymin, ymax = ax.get_yaxis().get_view_interval()
    ax.add_artist(
        matplotlib.lines.Line2D((xmin, xmax), (ymin, ymin), color="black", linewidth=2)
    )

    save_name = name.replace(" ", "_").lower()
    fig.savefig(
        os.path.join(post_process_dir, f"corruption_envelope_{save_name}.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
