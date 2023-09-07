import os
from typing import Optional
import torch
import pathlib
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
import numpy as np
from copy import deepcopy
import seaborn as sns
import pandas as pd

# sns.boxplot(
#     data=df, orient="v", x="System", y="NRMSE", hue="Name", ax=ax1, palette="Paired", linewidth=0.8
# )

np.random.seed(22)
torch.manual_seed(22)

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

current_dir = pathlib.Path(__file__).parent.resolve()

post_processing_path = os.path.join(
    current_dir, "results", "post_processing", "post_process.pt"
)
post_process_dict = torch.load(post_processing_path)
# exp_name = {0.01: "low_data", 0.25: "high_noise"}
# only ended up using high noise plot
exp_name = {0.25: "high_noise"}

svise_prefix = "svise_"
sr3_prefix = "sr3_"
stlsq_prefix = "stlsq_"
ens_prefix = "ens_"
uwb_prefix = "uwb_"

names = {
    svise_prefix: "SVISE",
    sr3_prefix: "SINDy-SR3",
    stlsq_prefix: "SINDy-STLSQ",
    ens_prefix: "ENS-SINDy",
    uwb_prefix: "WFB",
}

df = {"Method": [], "Value": [], "Metric_type": [], "Trial": [], "Noise": []}

def extract_from_key(key):
    method, noise, n_data, metric = key.split("_")[-4:]
    n_data = int(n_data)
    noise = float(noise.removesuffix("permille"))/1000
    metric = metric.upper()
    if metric == "MISMATCHED":
        metric = "NMT"
    method = names[method + "_"]
    system_name = "_".join(key.split("_")[:-4])
    return method, noise, n_data, metric, system_name

df_list = []
skip_list = []
for key, val in post_process_dict.items():
    method, noise, n_data, metric, system_name = extract_from_key(key)
    if (val == 1).any() and metric == "RER":
        skip_list.append([method, noise, n_data, system_name])
        continue
    elif [method, noise, n_data, system_name] in skip_list:
        continue
    df = dict(Value=val, Trial=np.arange(len(val)), Noise=noise, System=system_name, Metric_type=metric, Method=method)
    df = pd.DataFrame(df)
    df_list.append(df)
df = pd.concat(df_list).reset_index()

rer_max_values = df[df['Metric_type'] == 'RER'].groupby('Method')['Value'].max()
nmt_max_values = df[df['Metric_type'] == 'NMT'].groupby('Method')['Value'].max()

df.loc[(df['Metric_type'] == 'RER') & (df['Value'] == 1), 'Value'] = df['Method'].map(rer_max_values)
df["logValue"] = np.log10(df["Value"])

for noise, png_name in exp_name.items():
    fig, ax1 = plt.subplots(1, 1, figsize=convert_width((6, 2)) )
    ax = sns.boxplot(
        data=df[(df["Metric_type"] == "RER") & (df["Noise"] == noise)],
        y="Method",
        x="logValue",
        ax=ax1,
        palette="Set2",
        linewidth=0.8,
    )
    ax1.set_xlabel("$\log_{10}($" + "RER" + ")", color="k")
    ax.set(ylabel=None)
    fig.savefig(
        os.path.join(
            current_dir, "results", "post_processing", f"{png_name}_summary_rer.pdf"
        ),
        format="pdf",
        bbox_inches="tight",
    )

    fig, ax1 = plt.subplots(1, 1, figsize=convert_width((6, 2)) )
    ax = sns.boxplot(
        data=df[(df["Metric_type"] == "NMT") & (df["Noise"] == noise)],
        y="Method",
        x="Value",
        ax=ax1,
        palette="Set2",
        linewidth=0.8,
    )
    ax1.set_xlabel("Number of mismatched terms", color="k")
    ax.set(ylabel=None)
    fig.savefig(
        os.path.join(
            current_dir, "results", "post_processing", f"{png_name}_summary_nmd.pdf"
        ),
        format="pdf",
        bbox_inches="tight",
    )