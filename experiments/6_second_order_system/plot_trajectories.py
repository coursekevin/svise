import torch
from typing import Optional
from torch.distributions.normal import Normal
from svise.sde_learning import *
from svise.utils import *
from svise import odes
from scipy.integrate import solve_ivp
import pathlib
import os
import matplotlib.pyplot as plt
import matplotlib
import sys
import seaborn as sns

torch.manual_seed(2022)
np.random.seed(2022)
sys.setrecursionlimit(10000)
current_dir = pathlib.Path(__file__).parent.resolve()
post_processing_dir = os.path.join(current_dir, "results", "post_processing")
save_path = os.path.join(post_processing_dir, "traj_pred_posterior.pt")
torch.set_default_dtype(torch.float64)
model_dir = os.path.join("results", "models")

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


def load_model(file_path, d, var, G):
    n_reparam_samples = 512
    # settting up the model
    t0 = 0.0
    tf = 20.0
    input_labels = ["x_1", "x_2", "v_1", "v_2"]
    sde = SparsePolynomialIntegratorSDE(
        d,
        (t.min(), t.max()),
        degree=5,
        n_reparam_samples=n_reparam_samples,
        G=G,
        num_meas=G.shape[0],
        measurement_noise=var,
        input_labels=input_labels,
        n_tau=200,
    )
    sdict = torch.load(file_path)
    # adding for backwards compatibility
    sdict["marginal_sde.device_var"] = torch.empty(0)
    sde.load_state_dict(sdict)
    sde.eval()
    return sde


if __name__ == "__main__":
    recompute_posterior = False
    svise_prefix = "svise_"
    svise_paths = []
    for path, _, files in os.walk(os.path.join(current_dir, model_dir)):
        for f in files:
            if ".pt" in f and svise_prefix in f:
                svise_paths.append(os.path.join(current_dir, model_dir, f))
    dpath = "coupled_linear_100permille_0032data.pt"
    data_dict = torch.load(os.path.join(current_dir, "data", dpath))
    system_choice = data_dict["name"]
    d = data_dict["d"]
    num_data = data_dict["num_data"]
    t = data_dict["t"]
    G = data_dict["G"][:2]
    y_data = data_dict["y_data"][..., :2]
    var = data_dict["var"][:2]
    noise_percent = data_dict["noise_percent"]
    x0 = data_dict["x0"]
    degree = data_dict["degree"]
    system_name = data_dict["name"].replace(" ", "_").lower()
    model_name = f"svise_{system_name}_{noise_percent*1000:03.0f}permille_{num_data:04}"
    sde_model = [sdep for sdep in svise_paths if model_name in sdep]
    sde = load_model(sde_model[0], d, var, G)
    sde.sde_prior.resample_weights()
    sde.sde_prior.update_sparse_index()
    with open(os.path.join(post_processing_dir, "learnt_goveq.txt"), "w") as f:
            f.write("Learnt governing equations:\n")
            f.write(str(sde.sde_prior.feature_names))
    sde.sde_prior.reset_sparse_index()
    t0 = float(t.min())
    tf = float(t.max())
    W = 11
    H = 4
    num_H = 1000
    linewidth = 0.8
    skip_percent = 0.01
    t_eval = torch.linspace(t0, tf, 1000)
    sol = solve_ivp(odes.coupled_oscillator, (t.min(), t.max()), x0, t_eval=t_eval)
    y_eval = torch.tensor(sol.y.T, dtype=torch.float64)
    mu = sde.marginal_sde.mean(t_eval)
    covar = sde.marginal_sde.K(t_eval)
    var = covar.diagonal(dim1=-2, dim2=-1)
    var = var.clamp_min(1e-6)
    std = var.sqrt()
    lb = mu - 2 * std
    ub = mu + 2 * std
    d = mu.shape[-1]
    num_ts = len(t_eval)
    arrays = [mu, var, std, t_eval, y_eval]
    t_grid, y_grid = torch.meshgrid(
        t_eval,
        torch.linspace(y_eval.min() * 1.5, y_eval.max() * 1.5, num_H),
        indexing="ij",
    )
    fig, axs = plt.subplots(d, 1, figsize=convert_width((W, H), half_page=False), dpi=500, sharex=True)
    end = np.array([256, 256, 256, 0]) / 256
    start = matplotlib.colors.to_rgb("C0")
    start = [si for si in start]
    start.append(1.0)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "alpha_cmap", [end, start]
    )
    colors = [start, end]
    cmaps = ["Blues", "Blues"]
    num_plots = 200
    max_dist = 5.0
    norm = Normal(0, 1)
    with torch.no_grad():
        total_pdf = 0.0
        for k in range(mu.shape[-1]):
            log_like = -0.5 * (y_grid.t() - mu[:, k]).pow(2) / var[:, k]
            log_like += -0.5 * torch.log(2 * var[:, k] * math.pi)
            pdf = log_like.exp().t()
            pdf = (pdf.t() / pdf.max(dim=-1).values).t()
            pdf[pdf <= 0.5e-3] = 0.0
            if k == 0:
                axs[k].plot(
                    t_eval,
                    lb[:, k],
                    "C0-",
                    alpha=1.0,
                    linewidth=linewidth,
                    label="$2\sigma$",
                )
                axs[k].plot(t_eval, ub[:, k], "C0-", alpha=1.0, linewidth=linewidth)
                axs[k].fill_between(
                    t_eval,
                    lb[:, k],
                    y2=ub[:, k],
                    color="C0",
                    alpha=0.5,
                    linewidth=linewidth,
                )
            else:
                # axs[k].plot(t_eval, lb[:, k], "C0-", alpha=1.0, linewidth=linewidth)
                # axs[k].plot(t_eval, ub[:, k], "C0-", alpha=1.0, linewidth=linewidth)
                axs[k].fill_between(
                    t_eval,
                    lb[:, k],
                    y2=ub[:, k],
                    color="C0",
                    alpha=0.5,
                    linewidth=linewidth,
                )
            if k == 0:
                axs[k].plot(
                    t_eval,
                    y_eval[:, k],
                    "k--",
                    linewidth=linewidth,
                    label="True trajectory",
                )
            else:
                axs[k].plot(t_eval, y_eval[:, k], "k--", linewidth=linewidth)

        for k in range(y_data.shape[-1]):
            if k == 0:
                axs[k].plot(t, y_data[:, k], "kx", alpha=1.0, label="Data")
                # axs[k].legend()
            else:
                axs[k].plot(t, y_data[:, k], "kx", alpha=1.0)

        labels = ["$x_1(t)$", "$x_2(t)$", "$v_1(t)$", "$v_2(t)$"]
        for k in range(mu.shape[-1]):
            axs[k].set_frame_on(False)
            # ax.axis("off")
            axs[k].set_yticks([])
            xmin, xmax = axs[k].get_xaxis().get_view_interval()
            ymin, ymax = axs[k].get_yaxis().get_view_interval()
            if k == mu.shape[-1] - 1:
                axs[k].add_artist(
                    matplotlib.lines.Line2D(
                        (xmin, xmax), (ymin, ymin), color="black", linewidth=2
                    )
                )
                axs[k].set_xlabel("$t$ (sec)")
                ticks = np.linspace(t.min(), t.max(), 10)
            else:
                axs[k].set_xticks([])
            axs[k].set_ylabel(labels[k], rotation=0, labelpad=20)

    fig.savefig(
        os.path.join(post_processing_dir, f"traj_plot.pdf"),
        format="pdf",
        bbox_inches="tight",
    )

