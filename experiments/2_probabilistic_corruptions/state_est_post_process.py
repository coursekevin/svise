import os
import pathlib
import warnings

from tqdm import tqdm

from gpytorch.utils.warnings import GPInputWarning
import matplotlib.pyplot as plt
import numpy as np
from svise.sde_learning import *
from svise import odes
import torch
from svise.extern.gpytorch_fit import MultitaskGP



rs = 2022
torch.manual_seed(rs)
np.random.seed(rs)

warnings.filterwarnings("ignore", category=GPInputWarning)
torch.set_default_dtype(torch.float64)
# defining some assessment metrics


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


def rmse(m_pred, y_true, system_name):
    if system_name == "Hopf bifurcation".replace(" ", "_").lower():
        num_ignore = int(len(y_true) * 0.5)
        return (m_pred[num_ignore:] - y_true[num_ignore:]).pow(2).mean().sqrt()
    else:
        return (m_pred - y_true).pow(2).mean().sqrt()

def nrmse(m_pred, y_true, system_name):
    return rmse(m_pred, y_true, system_name) / rmse(torch.zeros_like(y_true), y_true, system_name)

# KL divergence is not meaningful for this case as the dynamics are now unknown


# get list of data files
current_dir = pathlib.Path(__file__).parent.resolve()
data_dir = "data"
data_paths = []
for path, _, files in os.walk(os.path.join(current_dir, data_dir)):
    for f in files:
        if ".pt" in f:
            data_paths.append(os.path.join(current_dir, data_dir, f))

# get list of all svise models
svise_prefix = "svise_"
dapper_prefix = "dapper_"
model_dir = os.path.join("results", "models")
svise_paths = []
dapper_paths = []
for path, _, files in os.walk(os.path.join(current_dir, model_dir)):
    for f in files:
        if ".pt" in f:
            if svise_prefix in f:
                svise_paths.append(os.path.join(current_dir, model_dir, f))
            elif dapper_prefix in f:
                dapper_paths.append(os.path.join(current_dir, model_dir, f))
systems = [
    "Damped linear oscillator",
    "Damped cubic oscillator",
    "Lorenz63",
    "Hopf bifurcation",
    "Selkov glycolysis model",
    "Duffing oscillator",
    "Coupled linear",
]
systems = [sys.replace(" ", "_").lower() for sys in systems]
columns = ["noise_percent", "num_data", "rmse", "kldiv"]
prefixes = [svise_prefix, dapper_prefix]
da_methods = ["PartFilt_",]
path_dict = {
    svise_prefix: svise_paths,
    dapper_prefix: dapper_paths,
}
post_process_dict = {
    f"{sys}_{prefix}{col}": []
    for prefix in prefixes
    for col in columns
    for sys in systems
}
post_process_dict = {}

pbar = tqdm(total=len(data_paths) + 1)
while data_paths:
    pbar.update(1)

    # loading data dictionary
    dpath = data_paths.pop()
    data_dict = torch.load(dpath)
    system_choice = data_dict["name"]
    d = data_dict["d"]
    W_true = data_dict["W"]
    noise_percent = data_dict["noise_percent"]
    num_data = data_dict["num_data"]
    t = data_dict["t"]
    y_true = data_dict["y_true"]
    y_data = data_dict["y_data"]
    G = data_dict["G"]
    var = data_dict["var"]
    K_true = var.diag()
    index = data_dict["index"]
    system_name = data_dict["name"].replace(" ", "_").lower()
    Q_diag = data_dict["pnoise_cov"]
    # --------------------------------------------------------------------------------------------------
    # SDE post-processing
    # loading model directory
    model_name = f"svise_{system_name}_{noise_percent*1000:03.0f}permille_{num_data:04}data_{index:02}"
    sde_model = [sdep for sdep in svise_paths if model_name in sdep]
    if len(sde_model) == 0:
        print("Model not found")
    elif len(sde_model) > 1:
        print("pause")
    # assert len(sde_model) == 1, "Found more than one model matching expression."
    model_dict = torch.load(sde_model[0])
    sde = SparsePolynomialSDE(
        d,
        (t.min(), t.max()),
        degree=5,
        n_reparam_samples=1024,
        G=G,
        num_meas=G.shape[0],
        measurement_noise=var,
    )
    sde.load_state_dict(model_dict)
    sde.eval()
    # saving to dictionary
    rmse_name = f"{system_name}_{svise_prefix}{noise_percent*1000:03.0f}permille_{num_data:04}_rmse"
    nrmse_name = f"{system_name}_{svise_prefix}{noise_percent*1000:03.0f}permille_{num_data:04}_nrmse"
    m_pred = sde.marginal_sde.mean(t)
    K_pred = sde.marginal_sde.K(t) + K_true
    post_process_dict.setdefault(rmse_name, []).append(rmse(m_pred, y_true, system_name))
    post_process_dict.setdefault(nrmse_name, []).append(nrmse(m_pred, y_true, system_name))
    # --------------------------------------------------------------------------------------------------
    # dapper post_processing
    model_name = f"{dapper_prefix}{system_name}_{noise_percent*1000:03.0f}permille_{num_data:04}data_{index:02}"
    model_path = [mp for mp in path_dict[dapper_prefix] if model_name in mp]
    assert len(model_path) == 1, "Found more than one model matching expression."
    da_dict = torch.load(model_path[0])
    for dam in da_methods:
        rmse_name = (
            f"{system_name}_{dam}{noise_percent*1000:03.0f}permille_{num_data:04}_rmse"
        )
        nrmse_name = (
            f"{system_name}_{dam}{noise_percent*1000:03.0f}permille_{num_data:04}_nrmse"
        )
        m_pred = torch.from_numpy(da_dict[f"{dam}mu"])
        post_process_dict.setdefault(rmse_name, []).append(rmse(m_pred, y_true, system_name))
        post_process_dict.setdefault(nrmse_name, []).append(nrmse(m_pred, y_true, system_name))



pbar.update(1)
pbar.close()
with torch.no_grad():
    for key, value in post_process_dict.items():
        post_process_dict[key] = torch.stack(value).numpy()
torch.save(
    post_process_dict,
    os.path.join(
        current_dir, "results", "post_processing", "state_est_post_process.pt"
    ),
)
