# this script takes in a list of data paths and runs some comparison models
from svise.sde_learning import *
from svise import odes
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader, TensorDataset
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pathlib
import argparse
from svise.extern.sindy_fit import SparsePolynomialSINDySTLSQ, SparsePolynomialSINDySR3, EnsembleSINDy

# from svise.extern.uwbayes_fit import SparsePolynomialUWBayes

save_dir = pathlib.Path(__file__).parent.resolve()

# random seed and dtype
rs = int(time.time())
torch.set_default_dtype(torch.float64)

parser = argparse.ArgumentParser(description="Comparison utility")
parser.add_argument("--dpath", type=str, help="data path")
# set random seed, otherwise use input
parser.add_argument("--rs", type=int, help="random seed", default=rs)
args = parser.parse_args()
rs = args.rs
save_path = os.path.join(save_dir, "results", "models")
torch.manual_seed(rs)
np.random.seed(rs)
# loading data
dpath = args.dpath
data_dict = torch.load(dpath)
system_choice = data_dict["name"]
d = data_dict["d"]
num_data = data_dict["num_data"]
t = data_dict["t"]
G = data_dict["G"]
y_data = data_dict["y_data"]
var = data_dict["var"]
index = data_dict["index"]
print(var)
noise_percent = data_dict["noise_percent"]
x0 = data_dict["x0"]
t0 = float(t.min())
tf = float(t.max())
# training stlsq
print_str = f"Running '{system_choice}' experiment with {num_data} data points and {var} Gaussian noise. Random seed {rs}."
print(print_str, flush=True)
print("training STLSQ.", flush=True)
ode = SparsePolynomialSINDySTLSQ(
    d, 5, t.numpy(), y_data.numpy()
)
system_name = system_choice.replace(" ", "_").lower()
fname = f"stlsq_{system_name}_{noise_percent*1000:03.0f}permille_{num_data:04}data_{index:02}_{rs}rs.pt"
torch.save(ode, os.path.join(save_path, fname))
# training sr3
print("training SR3.", flush=True)
ode = SparsePolynomialSINDySR3(d, 5, t.numpy(), y_data.numpy())
system_name = system_choice.replace(" ", "_").lower()
fname = f"sr3_{system_name}_{noise_percent*1000:03.0f}permille_{num_data:04}data_{index:02}_{rs}rs.pt"
torch.save(ode, os.path.join(save_path, fname))
# training ensemble sindy
# print("training ESINDY.", flush=True)
# ode = EnsembleSINDy(d, 5, t.numpy(), y_data.numpy())
# system_name = system_choice.replace(" ", "_").lower()
# fname = f"ens_{system_name}_{noise_percent*1000:03.0f}permille_{num_data:04}data_{index:02}_{rs}rs.pt"
# torch.save(ode, os.path.join(save_path, fname))
