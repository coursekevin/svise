from __future__ import annotations
import os
import math
from functools import partial
import random
import numpy as np
import torch
from torch import Tensor
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pickle as pkl
from typing import Tuple, Callable, Protocol
import pathlib

CURR_DIR = str(pathlib.Path(__file__).parent.absolute())
torch.set_default_dtype(torch.float64)


def second_order_fd(f_eval, dt):
    """Finite difference for second order derivative"""
    ddf_eval = np.zeros_like(f_eval)
    centered_diff = (f_eval[2:] - 2 * f_eval[1:-1] + f_eval[:-2]) / dt**2
    ddf_eval[1:-1] = centered_diff
    # Apply boundary conditions with one-sided difference
    ddf_eval[0] = (2 * f_eval[0] - 5 * f_eval[1] + 4 * f_eval[2] - f_eval[3]) / dt**2
    ddf_eval[-1] = (
        2 * f_eval[-1] - 5 * f_eval[-2] + 4 * f_eval[-3] - f_eval[-4]
    ) / dt**2
    return ddf_eval


class ExtremeMassBBH:
    good_initial_condition: np.ndarray = np.array([0.0, np.pi])

    def __init__(self, M: float = 1.0, e: float = 0.5, p: float = 100) -> None:
        self.M = M
        self.e = e
        self.p = p

    def __call__(self, t: float, state: np.ndarray) -> np.ndarray:
        """Extreme mass ratio binary black hole (equation 11 from:
        https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.3.043101)
        """
        M, e, p = self.M, self.e, self.p
        phi, chi = self.unpack_state(state)
        dphi = (p - 2 - 2 * e * np.cos(chi)) * (1 + e * np.cos(chi)) ** 2
        dphi /= M * p ** (3 / 2) * np.sqrt((p - 2) ** 2 - 4 * e**2)
        dchi = (
            (p - 2 - 2 * e * np.cos(chi))
            * (1 + e * np.cos(chi)) ** 2
            * np.sqrt(p - 6 - 2 * e * np.cos(chi))
        )
        dchi /= M * p**2 * np.sqrt((p - 2) ** 2 - 4 * e**2)
        return np.stack([dphi, dchi], axis=-1)

    def euclidean_norm(self, chi):
        M, e, p = self.M, self.e, self.p
        r = p * M / (1 + e * np.cos(chi))
        return r

    def convert_to_trajectories(self, state: np.ndarray) -> np.ndarray:
        phi, chi = self.unpack_state(state)
        r = self.euclidean_norm(chi)
        r_2 = np.stack([r * np.cos(phi), r * np.sin(phi)], axis=-1)
        return r_2

    @staticmethod
    def unpack_state(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return state[..., 0], state[..., 1]

    @classmethod
    def good_parameters(cls) -> ExtremeMassBBH:
        return cls(M=1.0, e=0.5, p=100)

    def get_trajectories(
        self, x0: np.ndarray, t: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        t_span = [t[0], t[-1]]
        sol = solve_ivp(
            self.__call__, t_span=t_span, y0=x0, t_eval=t, atol=1e-8, rtol=1e-6
        )
        return self.convert_to_trajectories(sol.y.T), sol.y.T

    def get_quadrupole(self, x0: np.ndarray, t: np.ndarray) -> np.ndarray:
        r = self.get_trajectories(x0, t)[0]
        x, y = r[..., 0], r[..., 1]
        Ixx = x**2 * self.M
        Iyy = y**2 * self.M
        Ixy = x * y * self.M
        return np.stack([Ixx, Iyy, Ixy], axis=-1)

    def wave_form(self, x0: np.ndarray, t: np.ndarray) -> np.ndarray:
        ddI = second_order_fd(self.get_quadrupole(x0, t), t[1] - t[0])
        ddIxx, ddIyy, ddIxy = ddI[..., 0], ddI[..., 1], ddI[..., 2]
        return (ddIxx - ddIyy) * math.sqrt(4 * math.pi / 5)


@dataclass(frozen=True)
class Data:
    train_t: Tensor
    train_y: Tensor  # grav. wave observations
    train_r: Tensor  # trajectories
    valid_t: Tensor
    valid_y: Tensor  # grav. wave observations
    valid_r: Tensor  # trajectories
    t_span: Tuple[float, float]
    scale_fn: Callable[[Tensor | float], Tensor]
    unscale_fn: Callable[[Tensor | float], Tensor]
    x0: Tensor
    valid_x0: Tensor

    def save(self, data_dir):
        with open(os.path.join(data_dir, "data.pkl"), "wb") as f:
            pkl.dump(self, f)

    @classmethod
    def load(cls, data_dir):
        with open(os.path.join(data_dir, "data.pkl"), "rb") as f:
            return pkl.load(f)


def scale(mu, scale, t):
    return (t - mu) / scale


def unscale(mu, scale, t):
    return t * scale + mu


def get_data(tend, ndata, noise) -> Data:
    multiplicative_scale = 5
    t_span = [0, tend]
    t_eval = np.linspace(0, tend * multiplicative_scale, ndata * multiplicative_scale)
    ode = ExtremeMassBBH.good_parameters()
    x0 = ode.good_initial_condition
    traj, state = ode.get_trajectories(x0, t_eval)
    x = torch.as_tensor(traj, dtype=torch.float64)
    t = torch.as_tensor(t_eval, dtype=torch.float64)
    wave_form = torch.as_tensor(ode.wave_form(x0, t_eval), dtype=torch.float64)
    wave_form = wave_form.unsqueeze(1)
    wave_form += torch.randn_like(wave_form) * noise
    # seperating into train and test
    train_ind = t_eval <= tend
    # mu = t_eval.mean()
    mu = torch.tensor(0.0)
    stdev = tend / 10
    scale_fn = partial(scale, mu, stdev)
    unscale_fn = partial(unscale, mu, stdev)
    data = Data(
        train_t=t[train_ind],
        train_y=wave_form[train_ind],
        train_r=x[train_ind],
        valid_t=t[~train_ind],
        valid_y=wave_form[~train_ind],
        valid_r=x[~train_ind],
        t_span=tuple(t_span),
        scale_fn=scale_fn,
        unscale_fn=unscale_fn,
        x0=torch.as_tensor(x0),
        valid_x0=torch.as_tensor(state[~train_ind][0]),
    )
    return data


def main():
    # get data
    random.seed(23)
    np.random.seed(23)
    tend, ndata, noise = 0.6e5, 1000, 1e-3

    data = get_data(tend, ndata, noise)
    data.save(CURR_DIR)
    print(f"Num. training data: {len(data.train_t)}")

    # plt.figure(figsize=(10, 5))
    # plt.plot(data.train_t, data.train_y, "-")
    # plt.plot(data.valid_t, data.valid_y, "--")
    # plt.savefig("wvf.png")

    # plt.plot(data.train_r[:, 0], data.train_r[:, 1], "-")
    # plt.plot(data.valid_r[:, 0], data.valid_r[:, 1], "-")
    # plt.axis("equal")



if __name__ == "__main__":
    main()
