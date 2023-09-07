import torch
from torch import Tensor
import math
import numpy as np
from typing import Union

__all__ = [
    "torch_ode",
    "lorenz63",
    "hopf_ode",
    "nonlinear_pendulum",
    "damped_oscillator",
    "selkov",
    "duffing",
    "coupled_oscillator",
    "lorenz96",
]


def torch_ode(t, x, ode, args=()):
    """Converts an ODE of generic form into a pytorch ODE

    Args:
      array: t: (..., )
      array: x: (... ,d)
      tuple: args: tuple of extra arguments to ode
      t: 
      x: 
      ode: 
      args:  (Default value = ())

    Returns:
      torch.tensor: derivative at (t, x) in a torch tensor

    """
    dx = torch.stack(ode(t, x, *args), -1)
    return dx


def lorenz63(t, x, sigma=10.0, beta=8 / 3, rho=28.0):
    """Provides the derivative for the Lorenz system at (x,t) given by,
    
        .. math::
    
            \begin{split}
                \dot{x} &= \sigma(y-x),\\
                \dot{y} &= x(\rho -z ) - y, \\
                \dot{z} &= xy - \beta z.
            \end{split}

    Args:
      np: ndarray x: (3,) state inputs
      float: t: time, not used by required by solve_ivp
      float: sigma: coefficient on dx[0]/dt
      float: beta: coefficient on dx[1]/dt
      float: rho: coefficient on dx[2]/dt
      t: 
      x: 
      sigma:  (Default value = 10.0)
      beta:  (Default value = 28.0)
      rho:  (Default value = 8 / 3)

    Returns:
      np.ndarray (3,): derivative at (x,t)

    """

    x_dot = sigma * (x[..., 1] - x[..., 0])
    y_dot = x[..., 0] * (rho - x[..., 2]) - x[..., 1]
    z_dot = x[..., 0] * x[..., 1] - beta * x[..., 2]

    return [x_dot, y_dot, z_dot]


def hopf_ode(t, x, mu=0.5, omega=1.0, A=1.0):
    """Computes and returns the hopf normal form derivative
    (models spontaneous oscillations in chemical reactions, electrical circuits, and fluid instability.)
    
    .. math::
    
        \begin{split}
            \dot{x} &= \mu x + \omega y - A x (x^2 + y^2),\\
            \dot{y} &= -\omega x + \mu y - A y (x^2 + y^2). \\
        \end{split}

    Args:
      np: ndarray x: state
      np: ndarray t: time (not used but required for solve_ivp )
      float: mu: bifurcation parameter
      float: omega: linear strength
      float: A: cubic strength
      t: 
      x: 
      mu:  (Default value = 0.6)
      omega:  (Default value = 1.0)
      A:  (Default value = 1.0)

    Returns:
      list: hopf normal form derivative

    """
    x_dot = (
        mu * x[..., 0]
        + omega * x[..., 1]
        - A * x[..., 0] * (x[..., 0] ** 2 + x[..., 1] ** 2)
    )
    y_dot = (
        -omega * x[..., 0]
        + mu * x[..., 1]
        - A * x[..., 1] * (x[..., 0] ** 2 + x[..., 1] ** 2)
    )
    return [x_dot, y_dot]


def nonlinear_pendulum(t, x, g=9.81, l=1.0, mu=0.35):
    """This function returns the nonlinear pendulum derivative

    Args:
      np: ndarray x: system state
      float: t: time (not used by function but required by solve_ivp)
      string: osc_type: flag indicating whether to return the linear / nonlinear derivative
      float: g: acceleration due to gravity
      float: l: length of rod
      float: mu: friction coefficient
      t: 
      x: 
      g:  (Default value = 9.81)
      l:  (Default value = 1.0)
      mu:  (Default value = 0.35)

    Returns:
      list: damped oscillator derivative

    """
    dx1 = x[..., 1]
    dx2 = -g / l * np.sin(x[..., 0]) - mu / l * x[..., 1]
    return [dx1, dx2]


def damped_oscillator(t, x, osc_type):
    """Computes and returns the damped oscilator derivative
    presented in "Discovering governing equations from data by sparse identification of nonlinear dynamical systems", Kutz et. al. 2016.
    If osc_type == 'linear' it returns the derivative for the linear system, otherwise it returns the derivate for the nonlinear system
    Linear system:
    
    .. math::
    
        \begin{split}
            \dot{x} &= -0.1x + 2y,\\
            \dot{y} &= -2x - 0.1y. \\
        \end{split}
    
    Nonlinear system:
    
    .. math::
    
        \begin{split}
            \dot{x} &= -0.1x^3 + 2y^3,\\
            \dot{y} &= -2x^3 - 0.1y^3. \\
        \end{split}

    Args:
      np: ndarray x: system state
      float: t: time (not used by function but required by solve_ivp)
      string: osc_type: flag indicating whether to return the linear / nonlinear derivative
      t: 
      x: 
      osc_type: 

    Returns:
      list: damped oscillator derivative

    """
    if osc_type == "linear":
        x_dot = -0.1 * x[..., 0] + 2 * x[..., 1]
        y_dot = -2 * x[..., 0] - 0.1 * x[..., 1]
    elif osc_type == "cubic":
        x_dot = -0.1 * x[..., 0] ** 3 + 2 * x[..., 1] ** 3
        y_dot = -2 * x[..., 0] ** 3 - 0.1 * x[..., 1] ** 3
    else:
        raise ValueError(f"Invalid osc_type: {osc_type}")

    return [x_dot, y_dot]


def selkov(t, x, a=0.08, b=0.6):
    """ODE model governing glycolysis. The Sel'kov model is given by,
    
    .. math::
    
        \begin{split}
            \dot{x} &= -x + ay + x^2 y,\\
            \dot{y} &= b - a y - x^2 y,\\
        \end{split}
    
    see Strogatz for more info.

    Args:
      list: x: system state (x[0] = ADP concentration, x[1] = F6P concentration)
      float: t: current time
      float: a: first kinetic parameter
      float: b: second kinetic parameter
      t: 
      x: 
      a:  (Default value = 0.08)
      b:  (Default value = 0.6)

    Returns:

    """
    xd = -x[..., 0] + a * x[..., 1] + x[..., 0] ** 2 * x[..., 1]
    yd = b - a * x[..., 1] - x[..., 0] ** 2 * x[..., 1]

    return [xd, yd]


def duffing(t, x, mu=0.35):
    """ODE model governing a damped highly nonlinear oscillator

    Args:
      t(float): time
      x(list system state): state
      mu(float, optional, optional): damping parameter, defaults to 0.35

    Returns:
      list: derivative at t, x

    """
    x1 = x[..., 0]
    x2 = x[..., 1]
    dx = x2
    dy = -(x1 ** 3 - x1) - mu * x2
    return [dx, dy]


def coupled_oscillator(
    t, z, k1=4, k2=2, k3=4, m1=1, m2=1.0, theta=math.pi * 0.0  # -11.0 / 180.0
):
    g = 9.81
    x1 = z[..., 0]
    x2 = z[..., 1]
    v1 = z[..., 2]
    v2 = z[..., 3]
    dz1 = v1
    dz2 = v2
    dz3 = -(k1 + k2) / m1 * x1 + k2 / m1 * x2 + g * math.sin(theta)
    dz4 = +k2 / m2 * x1 - (k2 + k3) / m2 * x2 + g * math.sin(theta)
    return [dz1, dz2, dz3, dz4]


def lorenz96(
    t, x: Union[np.ndarray, Tensor], F: float = 10.0
) -> Union[np.ndarray, Tensor]:
    r"""
    Lorenz 96 model.

    :param t: time
    :type t: float
    :param x: state
    :type x: list system state
    :param F: coupling coefficient, defaults to 10.0
    :type F: float, optional

    """
    if isinstance(x, np.ndarray):
        x_km2 = np.roll(x, 2, axis=-1)
        x_km1 = np.roll(x, 1, axis=-1)
        x_kp1 = np.roll(x, -1, axis=-1)
    elif isinstance(x, Tensor):
        x_km2 = x.roll(2, dims=-1)
        x_km1 = x.roll(1, dims=-1)
        x_kp1 = x.roll(-1, dims=-1)
    else:
        raise ValueError(f"Invalid x type: {type(x)}")
    return x_km1 * (x_kp1 - x_km2) - x + F


def roll(x, num: int):
    if isinstance(x, np.ndarray):
        return np.roll(x, num, axis=-1)
    elif isinstance(x, Tensor):
        return x.roll(num, dims=-1)
    else:
        raise ValueError(f"Invalid x type: {type(x)}")
