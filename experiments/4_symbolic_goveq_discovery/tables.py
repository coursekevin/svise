import os
import torch
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy

np.random.seed(22)
torch.manual_seed(22)

current_dir = pathlib.Path(__file__).parent.resolve()

post_processing_path = os.path.join(
    current_dir, "results", "post_processing", "post_process.pt"
)
post_process_dict = torch.load(post_processing_path)

systems = [
    "Damped linear oscillator",
    "Damped cubic oscillator",
    "Lorenz63",
    "Hopf bifurcation",
    "Selkov glycolysis model",
    "Duffing oscillator",
    "Coupled linear",
]
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
    "Damped linear oscillator": 2048,
    "Damped cubic oscillator": 2048,
    "Lorenz63": 2048,
    "Hopf bifurcation": 2048,
    "Selkov glycolysis model": 2048,
    "Duffing oscillator": 2048,
    "Coupled linear": 2048,
}
# low noise
joint_noise_dict = {0.01: low_noise_dict, 0.25: high_noise_dict}
exp_name = {0.01: "low data", 0.25: "high noise"}


svise_prefix = "svise_"
sr3_prefix = "sr3_"
stlsq_prefix = "stlsq_"
ens_prefix = "ens_"
uwb_prefix = "uwb_"
# prefix_list = [svise_prefix, uwb_prefix]
prefix_list = [svise_prefix, sr3_prefix, stlsq_prefix, ens_prefix]
rer_table_mean = {f"{prefix}rer_mean": [] for prefix in prefix_list}
rer_table_var = {f"{prefix}rer_var": [] for prefix in prefix_list}
rer_table = {**rer_table_mean, **rer_table_var}
mmd_table_mean = {f"{prefix}mmd_mean": [] for prefix in prefix_list}
mmd_table_var = {f"{prefix}mmd_var": [] for prefix in prefix_list}
failure_table = {f"{prefix}failure": [] for prefix in prefix_list}
mmd_table = {**mmd_table_mean, **mmd_table_var}
low_noise_table = {**rer_table, **mmd_table, **failure_table}
high_noise_table = deepcopy(low_noise_table)

# for noise_percent, noise_dict in joint_noise_dict.items():
noise_percent, noise_dict = 0.01, low_noise_dict
for experiment in systems:
    system_name = experiment.replace(" ", "_").lower()
    num_data = noise_dict[experiment]
    disp_data = []
    for prefix in prefix_list:
        rer_name = f"{system_name}_{prefix}{noise_percent*1000:03.0f}permille_{num_data:04}_rer"
        mmd_name = f"{system_name}_{prefix}{noise_percent*1000:03.0f}permille_{num_data:04}_mismatched"
        disp_data.append(post_process_dict[rer_name])
        low_noise_table[f"{prefix}rer_mean"].append(post_process_dict[rer_name].mean())
        low_noise_table[f"{prefix}rer_var"].append(post_process_dict[rer_name].var())
        low_noise_table[f"{prefix}failure"].append(
            (post_process_dict[rer_name] == 1).all()
        )
        low_noise_table[f"{prefix}mmd_mean"].append(post_process_dict[mmd_name].mean())
        low_noise_table[f"{prefix}mmd_var"].append(post_process_dict[mmd_name].var())
    # ax.boxplot(disp_data)
    # ax.set_ylabel(f"{experiment}, {exp_name[noise_percent]}")
# print(low_noise_table)
for key, val in low_noise_table.items():
    assert len(val) == len(systems), "Length of data is not equal to number of systems"
num_stds = 1.0
row = "System &"
for i, prefix in enumerate(prefix_list):
    prefix = prefix.replace("_", " ")
    # row += f"{prefix} RER & {prefix} MMD"
    row += f"{prefix} RER"
    if i == len(prefix_list) - 1:
        row += "\\\ \n"
    else:
        row += " & "
for i, system in enumerate(systems):
    row += f"{system} & "
    for j, prefix in enumerate(prefix_list):
        rer_mean = low_noise_table[f"{prefix}rer_mean"][i]
        rer_var = low_noise_table[f"{prefix}rer_var"][i]
        mmd_mean = low_noise_table[f"{prefix}mmd_mean"][i]
        mmd_var = low_noise_table[f"{prefix}mmd_var"][i]
        if low_noise_table[f"{prefix}failure"][i]:
            indicator = "^*"
        else:
            indicator = ""

        if j == 0:
            row += f"$\\mathbf{{{rer_mean:.3f} \pm {num_stds * np.sqrt(rer_var):.3f}}}{indicator}$"  # & "
        else:
            if not low_noise_table[f"{prefix}failure"][i]:
                row += f"${rer_mean:.3f} \pm {num_stds * np.sqrt(rer_var):.3f}{indicator}$"  # & "
            else:
                row += f"--"  # & "
        # row += f"${mmd_mean:.3f} \pm {num_stds * np.sqrt(mmd_var):.3f}$"
        if j == len(prefix_list) - 1:
            row += "\\\ \n"
        else:
            row += " & "

table_path = os.path.join(
    current_dir, "results", "post_processing", "low_noise_rer_table.txt"
)
with open(table_path, "w") as f:
    f.write(row)

row = "System &"
for i, prefix in enumerate(prefix_list):
    prefix = prefix.replace("_", " ")
    # row += f"{prefix} RER & {prefix} MMD"
    row += f"{prefix} RER"
    if i == len(prefix_list) - 1:
        row += "\\\ \n"
    else:
        row += " & "
for i, system in enumerate(systems):
    row += f"{system} & "
    for j, prefix in enumerate(prefix_list):
        rer_mean = low_noise_table[f"{prefix}rer_mean"][i]
        rer_var = low_noise_table[f"{prefix}rer_var"][i]
        mmd_mean = low_noise_table[f"{prefix}mmd_mean"][i]
        mmd_var = low_noise_table[f"{prefix}mmd_var"][i]
        if low_noise_table[f"{prefix}failure"][i]:
            indicator = "^*"
        else:
            indicator = ""
        if j == 0:
            # row += f"$\\textbf{{{rer_mean:.3f} \pm {num_stds * np.sqrt(rer_var):.3f}}}$"  # & "
            row += f"$\\mathbf{{{mmd_mean:.3f} \pm {num_stds * np.sqrt(mmd_var):.3f}}}{indicator}$"
        else:
            if not low_noise_table[f"{prefix}failure"][i]:
                row += f"${mmd_mean:.3f} \pm {num_stds * np.sqrt(mmd_var):.3f}{indicator}$"  # & "
            else:
                row += f"--"  # & "
        if j == len(prefix_list) - 1:
            row += "\\\ \n"
        else:
            row += " & "

table_path = os.path.join(
    current_dir, "results", "post_processing", "low_noise_mmd_table.txt"
)
with open(table_path, "w") as f:
    f.write(row)

noise_percent, noise_dict = 0.25, high_noise_dict
for experiment in systems:
    system_name = experiment.replace(" ", "_").lower()
    num_data = noise_dict[experiment]
    disp_data = []
    for prefix in prefix_list:
        rer_name = f"{system_name}_{prefix}{noise_percent*1000:03.0f}permille_{num_data:04}_rer"
        mmd_name = f"{system_name}_{prefix}{noise_percent*1000:03.0f}permille_{num_data:04}_mismatched"
        disp_data.append(post_process_dict[rer_name])
        high_noise_table[f"{prefix}rer_mean"].append(post_process_dict[rer_name].mean())
        high_noise_table[f"{prefix}rer_var"].append(post_process_dict[rer_name].var())
        high_noise_table[f"{prefix}failure"].append(
            (post_process_dict[rer_name] == 1).all()
        )
        high_noise_table[f"{prefix}mmd_mean"].append(post_process_dict[mmd_name].mean())
        high_noise_table[f"{prefix}mmd_var"].append(post_process_dict[mmd_name].var())
    # ax.boxplot(disp_data)
    # ax.set_ylabel(f"{experiment}, {exp_name[noise_percent]}")
# print(high_noise_table)
for key, val in high_noise_table.items():
    assert len(val) == len(
        systems
    ), f"length of data {len(val)} is not equal to number of systems {len(systems)}"
# num_stds = 2.0
row = "System &"
for i, prefix in enumerate(prefix_list):
    prefix = prefix.replace("_", " ")
    # row += f"{prefix} RER & {prefix} MMD"
    row += f"{prefix} RER"
    if i == len(prefix_list) - 1:
        row += "\\\ \n"
    else:
        row += " & "
for i, system in enumerate(systems):
    row += f"{system} & "
    for j, prefix in enumerate(prefix_list):
        rer_mean = high_noise_table[f"{prefix}rer_mean"][i]
        rer_var = high_noise_table[f"{prefix}rer_var"][i]
        mmd_mean = high_noise_table[f"{prefix}mmd_mean"][i]
        mmd_var = high_noise_table[f"{prefix}mmd_var"][i]
        if high_noise_table[f"{prefix}failure"][i]:
            indicator = "^*"
        else:
            indicator = ""

        if j == 0:
            row += f"$\\mathbf{{{rer_mean:.3f} \pm {num_stds * np.sqrt(rer_var):.3f}}}{indicator}$"  # & "
        else:
            if not high_noise_table[f"{prefix}failure"][i]:
                row += f"${rer_mean:.3f} \pm {num_stds * np.sqrt(rer_var):.3f}{indicator}$"  # & "
            else:
                row += f"--"  # & "
        # row += f"${mmd_mean:.3f} \pm {num_stds * np.sqrt(mmd_var):.3f}$"
        if j == len(prefix_list) - 1:
            row += "\\\ \n"
        else:
            row += " & "

table_path = os.path.join(
    current_dir, "results", "post_processing", "high_noise_rer_table.txt"
)
with open(table_path, "w") as f:
    f.write(row)

row = "System &"
for i, prefix in enumerate(prefix_list):
    prefix = prefix.replace("_", " ")
    # row += f"{prefix} RER & {prefix} MMD"
    row += f"{prefix} RER"
    if i == len(prefix_list) - 1:
        row += "\\\ \n"
    else:
        row += " & "
for i, system in enumerate(systems):
    row += f"{system} & "
    for j, prefix in enumerate(prefix_list):
        rer_mean = high_noise_table[f"{prefix}rer_mean"][i]
        rer_var = high_noise_table[f"{prefix}rer_var"][i]
        mmd_mean = high_noise_table[f"{prefix}mmd_mean"][i]
        mmd_var = high_noise_table[f"{prefix}mmd_var"][i]
        if high_noise_table[f"{prefix}failure"][i]:
            indicator = "^*"
        else:
            indicator = ""
        if j == 0:
            # row += f"$\\textbf{{{rer_mean:.3f} \pm {num_stds * np.sqrt(rer_var):.3f}}}$"  # & "
            row += f"$\\mathbf{{{mmd_mean:.3f} \pm {num_stds * np.sqrt(mmd_var):.3f}}}{indicator}$"
        else:
            if not high_noise_table[f"{prefix}failure"][i]:
                row += f"${mmd_mean:.3f} \pm {num_stds * np.sqrt(mmd_var):.3f}{indicator}$"  # & "
            else:
                row += f"--"  # & "
        if j == len(prefix_list) - 1:
            row += "\\\ \n"
        else:
            row += " & "

table_path = os.path.join(
    current_dir, "results", "post_processing", "high_noise_mmd_table.txt"
)
with open(table_path, "w") as f:
    f.write(row)
