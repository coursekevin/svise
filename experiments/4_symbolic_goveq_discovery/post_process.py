from svise.sde_learning import *
import os
import pathlib
import torch

torch.manual_seed(22)


def rer(W_true, W):
    # reconstruction error
    if not torch.is_tensor(W):
        W = torch.tensor(W)
    return (W - W_true).pow(2).sum().sqrt() / W_true.pow(2).sum().sqrt()


def num_mismatched(W_true, W):
    # compute number of mismatched entries
    if not torch.is_tensor(W):
        W = torch.tensor(W)
    W_true_ind = torch.zeros_like(W_true)
    W_true_ind[W_true != 0] = 1.0
    W_ind = torch.zeros_like(W)
    W_ind[W != 0] = 1.0
    return (W_true_ind - W_ind).abs().sum()


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
sr3_prefix = "sr3_"
stlsq_prefix = "stlsq_"
ens_prefix = "ens_"
model_dir = os.path.join("results", "models")
svise_paths = []
sr3_paths = []
stlsq_paths = []
ens_paths = []
for path, _, files in os.walk(os.path.join(current_dir, model_dir)):
    for f in files:
        if ".pt" in f:
            if svise_prefix in f:
                svise_paths.append(os.path.join(current_dir, model_dir, f))
            elif sr3_prefix in f:
                sr3_paths.append(os.path.join(current_dir, model_dir, f))
            elif stlsq_prefix in f:
                stlsq_paths.append(os.path.join(current_dir, model_dir, f))
            elif ens_prefix in f:
                ens_paths.append(os.path.join(current_dir, model_dir, f))
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
columns = ["noise_percent", "num_data", "rer", "num_mismatched"]
prefixes = [svise_prefix, sr3_prefix, stlsq_prefix, ens_prefix]
path_dict = {
    svise_prefix: svise_paths,
    sr3_prefix: sr3_paths,
    stlsq_prefix: stlsq_paths,
    ens_prefix: ens_paths,
}
post_process_dict = {
    f"{sys}_{prefix}{col}": []
    for prefix in prefixes
    for col in columns
    for sys in systems
}
post_process_dict = {}

while data_paths:
    # loading data dictionary
    dpath = data_paths.pop()
    data_dict = torch.load(dpath)
    d = data_dict["d"]
    W_true = data_dict["W"]
    noise_percent = data_dict["noise_percent"]
    num_data = data_dict["num_data"]
    t = data_dict["t"]
    G = data_dict["G"]
    var = data_dict["var"]
    index = data_dict["index"]
    system_name = data_dict["name"].replace(" ", "_").lower()
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
    sde.resample_sde_params()
    W = sde.sde_prior._tfm(sde.sde_prior.W.mean(0))
    # saving to dictionary
    rer_name = f"{system_name}_{svise_prefix}{noise_percent*1000:03.0f}permille_{num_data:04}_rer"
    num_mmd_name = f"{system_name}_{svise_prefix}{noise_percent*1000:03.0f}permille_{num_data:04}_mismatched"
    post_process_dict.setdefault(rer_name, []).append(rer(W_true, W))
    post_process_dict.setdefault(num_mmd_name, []).append(num_mismatched(W_true, W))
    # post-processing of linear estimation schemes
    for prefix in prefixes[1:]:
        assert (
            prefix != svise_prefix
        ), "this post processing is only intended linear models"
        model_name = f"{prefix}{system_name}_{noise_percent*1000:03.0f}permille_{num_data:04}data_{index:02}"
        model_path = [mp for mp in path_dict[prefix] if model_name in mp]
        assert len(model_path) == 1, "Found more than one model matching expression."
        model = torch.load(model_path[0])
        rer_name = f"{system_name}_{prefix}{noise_percent*1000:03.0f}permille_{num_data:04}_rer"
        num_mmd_name = f"{system_name}_{prefix}{noise_percent*1000:03.0f}permille_{num_data:04}_mismatched"
        post_process_dict.setdefault(rer_name, []).append(rer(W_true, model.W))
        post_process_dict.setdefault(num_mmd_name, []).append(
            num_mismatched(W_true, model.W)
        )
with torch.no_grad():
    for key, value in post_process_dict.items():
        post_process_dict[key] = torch.stack(value).numpy()
torch.save(
    post_process_dict,
    os.path.join(current_dir, "results", "post_processing", "post_process.pt"),
)
