from typing import Optional
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import pathlib
import os
import seaborn as sns
import pandas as pd

np.random.seed(22)
torch.manual_seed(22)

# torch.set_default_dtype(torch.float64)
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

def convert_width(fsize: tuple[float, float], half_page: Optional[bool] = True) -> tuple[float, float]:
    rescale_width = WIDTH_INCHES if half_page else 2 * WIDTH_INCHES
    width = fsize[0] 
    return tuple(size * rescale_width / width for size in fsize)

current_dir = pathlib.Path(__file__).parent.resolve()
post_process_dir = os.path.join(
    current_dir,
    "results",
    "post_processing",
)
post_process_dict = torch.load(os.path.join(post_process_dir, "post_process.pt"))
plotting_dict = {
    "rank": [],
    "log10rer": [],
    "nmm": [],
}
data_frames = []
for rank, value in post_process_dict.items():
    tmp_df = pd.DataFrame()
    tmp_df["log10rer"] = np.log10(value["RER"])
    tmp_df["nmm"] = value["NMM"]
    tmp_df["rank"] = rank
    data_frames.append(tmp_df)
plotting_df = pd.concat(data_frames)
fig, ax = plt.subplots(1, 1, figsize=convert_width((5, 5)))
ax = sns.boxplot(
    data=plotting_df[plotting_df["rank"] >= 256],
    x="rank",
    y="log10rer",
    ax=ax,
    color=sns.color_palette("Set2").as_hex()[0],
)
# ax.set_xticklabels(plotting_dict["rank"])
ax.set_xlabel("Measurement matrix rank")
ax.set_ylabel("$\log_{10}($" + "RER" + ")", color="k")
# ax.grid()
fig.savefig(
    os.path.join(post_process_dir, f"rank.pdf"),
    format="pdf",
    bbox_inches="tight",
)
fig, ax = plt.subplots(1, 1, figsize=convert_width((5, 4)))
ax = sns.boxplot(
    data=plotting_df[plotting_df["rank"] >= 256],
    x="rank",
    y="nmm",
    ax=ax,
    color=sns.color_palette("Set2").as_hex()[0],
    linewidth=0.8,
)
ax.set_xlabel("Measurement matrix rank")
ax.set_ylabel("Number of mismatched terms", color="k")
# ax.grid()
fig.savefig(
    os.path.join(post_process_dir, f"nmm.pdf"),
    format="pdf",
    bbox_inches="tight",
)

fig = plt.figure(figsize=convert_width((11, 4.0), half_page=False))
dpath = os.path.join(current_dir, "")
dpath = os.path.join(current_dir, "data", "lorenz96_rank0512_5.pt")
data = torch.load(dpath)
plt.contourf(
    np.arange(1024),
    data["t"][data["t"] >= 1.0],
    data["y_true"][data["t"] >= 1.0, :],
    cmap=sns.color_palette("Spectral", as_cmap=True),
)
plt.xlabel("State index")
plt.ylabel("Time (days)")
fig.savefig(
    os.path.join(post_process_dir, f"l96plot.pdf"),
    format="pdf",
    bbox_inches="tight",
)