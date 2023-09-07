import os
import torch
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
from numpy import log10, concatenate
from copy import deepcopy
from tqdm import tqdm
import seaborn as sns

sns.set(font_scale=2)
current_dir = pathlib.Path(__file__).parent.resolve()

post_processing_path = os.path.join(
    current_dir, "results", "post_processing", "state_est_post_process.pt"
)
sns.set(style="ticks")
mpl.rc("text", usetex=True)
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

post_process_dict = torch.load(post_processing_path)

# amount of data for low noise regime
low_noise_dict = {
    "Damped linear oscillator": 16,
    "Damped cubic oscillator": 16,
    "Lorenz63": 64,
    "Hopf bifurcation": 32,
    "Selkov glycolysis model": 32,
    "Duffing oscillator": 32,
    "Coupled linear": 64,
}
# data for high noise regime
high_noise_dict = {
    "Damped linear oscillator": 128,
    "Damped cubic oscillator": 128,
    "Lorenz63": 128,
    "Hopf bifurcation": 128,
    "Selkov glycolysis model": 128,
    "Duffing oscillator": 128,
    "Coupled linear": 128,
}
# low noise
joint_noise_dict = {0.01: low_noise_dict, 0.25: high_noise_dict}
exp_name = {0.01: "low data", 0.25: "high noise"}

# don't plot GP because it requires a lot of explanation to understand why it will
# sometimes outperform the standard SE techniques when dynamics are incorrect
# prefix_list = ["svise_", "ExtKF_", "iEnKS_", "Var4D_", "PartFilt_"]
# name_list = ["SVISE", "ExtKF", "iEnKS", "Var4D", "PF"]
# should try tuning var4d and iEnKS a bit better
prefix_list = ["svise_", "PartFilt_"]
name_list = ["SVISE", "PF"]

# for settings in ((0.25, high_noise_dict), (0.01, low_noise_dict)):
settings = (0.1, high_noise_dict)
noise_percent, noise_dict = settings

system_list = [
    "Coupled linear", # no need for linear forward model
    # "Damped linear oscillator",  # no need linear forward model
    "Damped cubic oscillator",
    "Lorenz63",
    "Hopf bifurcation", # difficult to get initial condition correct for other methods
    "Selkov glycolysis model",
    "Duffing oscillator",
]

system_plot_name = {
    "Coupled linear": "Coupled linear", # no need for linear forward model
    # "Damped linear oscillator": "Linear osc.",  # no need linear forward model
    "Damped cubic oscillator": "Cubic osc.",
    "Lorenz63": "Lorenz `63",
    "Hopf bifurcation": "Hopf bifurcation", # difficult to get initial condition correct for other methods
    # "Hopf bifurcation", # difficult to get initial condition correct for other methods
    "Selkov glycolysis model": "Selkov",
    "Duffing oscillator": "Duffing osc.",
}

df = {"System": system_list}
df = {"System": [], "Name": [], "NRMSE": [], "Trial": []}
fig, ax1 = plt.subplots(1, 1, figsize=convert_width((8, 5), page_scale=0.7))
for prefix, name in zip(prefix_list, name_list):
    nrmse = []
    for experiment in system_list:
        system_name = experiment.replace(" ", "_").lower()
        num_data = noise_dict[experiment]
        nrmse_name = f"{system_name}_{prefix}{noise_percent*1000:03.0f}permille_{num_data:04}_nrmse"
        for trial, nrmse in enumerate(np.log10(post_process_dict[nrmse_name])):
            df["System"].append(system_plot_name[experiment])
            df["Name"].append(name)
            df["NRMSE"].append(nrmse)
            df["Trial"].append(trial)

sns.boxplot(
    data=df, y="System", x="NRMSE", hue="Name", ax=ax1, palette="Set2", linewidth=0.8
)
ax1.set_xlabel("$\log_{10}($" + "NRMSE" + ")", color="k", fontweight="bold")
ax1.set_xlim((-2.131494963472689, -0.17461025109003986))

# plt.show()

fig.savefig(
    os.path.join(
        current_dir,
        "results",
        "post_processing",
        f"pure_state_est_summary.pdf",
    ),
    format="pdf",
    bbox_inches="tight",
)
