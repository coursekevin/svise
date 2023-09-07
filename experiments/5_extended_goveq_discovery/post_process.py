from svise.sde_learning import *
from svise.variationalsparsebayes.sparse_glm import SparsePolynomialNeighbour1D
from svise import quadrature
import os
import pathlib
import torch
import numpy as np

torch.manual_seed(22)
np.random.seed(22)


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


def load_model(t, degree, G, var):
    n_reparam_samples = 1024
    Q_diag = torch.ones(d) * 1e-0
    diffusion_prior = SparseDiagonalDiffusionPrior(d, Q_diag, n_reparam_samples, 1e-5)
    # getting a good guess for the initial state
    t0 = float(t.min())
    tf = float(t.max())
    marginal_sde = DiagonalMarginalSDE(
        d,
        (t0, tf),
        diffusion_prior=diffusion_prior,
        model_form="GLM",
        n_tau=200,
        learn_inducing_locations=False,
    )
    num_meas = G.shape[0]
    quad_scheme = quadrature.UnbiasedGaussLegendreQuad(t0, tf, 128, quad_percent=0.8)
    likelihood = IndepGaussLikelihood(G, num_meas, var)
    features = SparsePolynomialNeighbour1D(d, degree=degree, input_labels=["x"])
    sde_prior = SparseNeighbourGLM(
        d,
        SparseFeatures=features,
        n_reparam_samples=n_reparam_samples,
        tau=1e-5,
    )
    sde = SDELearner(
        marginal_sde=marginal_sde,
        likelihood=likelihood,
        quad_scheme=quad_scheme,
        sde_prior=sde_prior,
        diffusion_prior=diffusion_prior,
        n_reparam_samples=n_reparam_samples,
    )
    sde.train()
    return sde


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
model_dir = os.path.join("results")
svise_paths = []
for path, _, files in os.walk(os.path.join(current_dir, model_dir)):
    for f in files:
        if ".pt" in f:
            if svise_prefix in f:
                svise_paths.append(os.path.join(current_dir, model_dir, f))
systems = ["Lorenz96"]
systems = [sys.replace(" ", "_").lower() for sys in systems]
columns = ["noise_percent", "num_data", "rer", "num_mismatched"]
prefixes = [svise_prefix]
path_dict = {svise_prefix: svise_paths}
post_process_dict = {
    f"{sys}_{prefix}{col}": []
    for prefix in prefixes
    for col in columns
    for sys in systems
}
rank_list = 2 ** np.array([6, 7, 8, 9, 10])
rer_name = f"RER"
nmm_name = f"NMM"
post_process_dict = {rank: {} for rank in rank_list}
while data_paths:
    # loading data dictionary
    dpath = data_paths.pop()
    exp_num = int(dpath.split("_")[-1][:-3])
    data_dict = torch.load(dpath)
    d = data_dict["d"]
    W_true = data_dict["W"]
    noise_percent = data_dict["noise_percent"]
    num_data = data_dict["num_data"]
    t = data_dict["t"]
    G = data_dict["G"]
    var = data_dict["var"]
    rank = data_dict["measurement_rank"]
    G = data_dict["G"]
    system_name = data_dict["name"].replace(" ", "_").lower()
    # SDE post-processing
    # loading model directory
    model_name = f"svise_{system_name}_rank{rank}_{exp_num:02}"
    sde_model = [sdep for sdep in svise_paths if model_name in sdep]
    if len(sde_model) == 0:
        print(f"Model {dpath} not found")
        continue
    elif len(sde_model) > 1:
        print("More than one result matching model found...")
    # assert len(sde_model) == 1, "Found more than one model matching expression."
    model_dict = torch.load(sde_model[0])
    sde = load_model(t, 2, G, var)
    sde.load_state_dict(model_dict)
    sde.eval()
    sde.resample_sde_params()
    W = sde.sde_prior._tfm(sde.sde_prior.W.mean(0))
    post_process_dict[rank].setdefault(rer_name, []).append(rer(W_true, W))
    post_process_dict[rank].setdefault(nmm_name, []).append(num_mismatched(W_true, W))
# saving to dictionary
with torch.no_grad():
    for rank in rank_list:
        for key in [rer_name, nmm_name]:
            post_process_dict[rank][key] = torch.stack(
                post_process_dict[rank][key]
            ).numpy()
torch.save(
    post_process_dict,
    os.path.join(current_dir, "results", "post_processing", "post_process.pt"),
)
