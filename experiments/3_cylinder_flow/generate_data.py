from __future__ import annotations
from dataclasses import dataclass
import os
import pathlib
import pickle as pkl

import torch
from torch import Tensor
import numpy as np

from svise import pca

CURR_DIR = str(pathlib.Path(__file__).parent.resolve())
DATA_PATH = os.path.join(CURR_DIR, "data", "vortex.pkl")
DATA_DIR = os.path.join(CURR_DIR, "data")


@dataclass(frozen=True)
class FlowDataContainer:
    """Container for flow data"""

    t: Tensor
    x: Tensor
    y: Tensor
    vel_x: Tensor
    vel_y: Tensor
    train_y: Tensor
    vorticity: Tensor


@dataclass(frozen=True)
class CylinderFlowData:
    """storage container for raw cylinder flow data"""
    train_data: FlowDataContainer
    test_data: FlowDataContainer
    grid_shape: tuple[int, int] = (199, 1499)
    adjusted_grid: tuple[int ,int] = (199, 1499)

    @classmethod
    def load_from_file(
        cls, file_path: str, skip_percent: float = 0.2, test_percent: float = 0.2, dtype: torch.dtype = torch.float64
    ) -> CylinderFlowData:
        """loads a cylinder dataset from a file_path"""
        with open(file_path, "rb") as handle:
            data = pkl.load(handle)

        def filter_by_name(name):
            flattened_data = torch.tensor(data["x"][..., data["var_names"].index(name)], dtype=dtype)
            batch_size = flattened_data.shape[0]
            flattened_data = flattened_data.reshape(batch_size, *cls.grid_shape)[:, :, :cls.adjusted_grid[1]]
            return flattened_data.reshape(batch_size, -1)

        assert 0 <= skip_percent < 1, "skip must be in range [0, 1)"

        def remove_percent(data):
            # the first couple of frames have some transients that
            # are highly nonlinear, let's just ignore them for now
            skip = int(len(data) * skip_percent)
            return data[skip:]

        def get_train(data):
            data = remove_percent(data)
            test = int(len(data) * test_percent)
            return data[:-test]

        def get_test(data):
            data = remove_percent(data)
            test = int(len(data) * test_percent)
            return data[-test:]

        def stack_velocity():
            u = filter_by_name("u")
            v = filter_by_name("v")
            return torch.cat([u, v], dim=1)

        t = torch.tensor([0.1 * j for j in range(data["x"].shape[0])])
        return cls(
            train_data=FlowDataContainer(
                t=get_train(t),
                x=get_train(filter_by_name("x")),
                y=get_train(filter_by_name("y")),
                vel_x=get_train(filter_by_name("u")),
                vel_y=get_train(filter_by_name("v")),
                train_y=get_train(stack_velocity()),
                vorticity=get_train(filter_by_name("Vorticity")),
            ),
            test_data=FlowDataContainer(
                t=get_test(t),
                x=get_test(filter_by_name("x")),
                y=get_test(filter_by_name("y")),
                vel_x=get_test(filter_by_name("u")),
                vel_y=get_test(filter_by_name("v")),
                train_y=get_test(stack_velocity()),
                vorticity=get_test(filter_by_name("Vorticity")),
            ),
        )


def main():
    rs = 21
    torch.manual_seed(rs)
    np.random.seed(rs)
    print("Loading data...")
    data = CylinderFlowData.load_from_file(DATA_PATH, dtype=torch.float32)
    # ---------------------- PCA ----------------------
    train_y = data.train_data.train_y
    print("Performing PCA...")
    lin_model, z = pca.PCA.create(train_y, percent_cutoff=0.9, max_evecs=100)
    # assume 0.001 % error in code vectors.
    code_stdev = torch.ones_like(z[0]) * 1e-3
    valid_y = data.test_data.train_y
    x_grid = data.train_data.x[0].reshape(*data.adjusted_grid)[0]
    y_grid = data.train_data.y[0].reshape(*data.adjusted_grid)[:, 0]
    data = {
        "z": z,
        "code_stdev": code_stdev,
        "lin_model": lin_model.state_dict(),
        "t": data.train_data.t,
        "valid_t": data.test_data.t,
        "valid_z": lin_model.encode(valid_y),
        "grid": (x_grid, y_grid),
        "valid_y": data.test_data.train_y,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "encoded_data.pkl"), "wb") as f:
        pkl.dump(data, f)
    print(f"Done, found {data['z'].shape[1]} modes.")

if __name__ == "__main__":
    main()
