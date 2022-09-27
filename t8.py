import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

def f(t: npt.ArrayLike, x: npt.ArrayLike)->npt.ArrayLike:
    x: npt.ndarray = np.array(x)
    return np.array([998 * x[0] + 1998 * x[1], -999 * x[0] - 1999 * x[1]])


def Newton(f)->npt.ArrayLike:
    """Newton method for n-dimensional system

    Args:
        f (Any): function

    Returns:
        npt.ArrayLike: solution
    """
    

def RK2(func, t0: float, x0: npt.ArrayLike, t: float, alpha: float = 3 / 4)->npt.ArrayLike:
    """ The second order Runge-Kutta method

    Args:
        func (Any): function f in dx/dt = f(t, x)
        t0 (float): point of initial value
        x0 (float): initial value x(t0) = x0
        t (float): point where to evaluate solution x(t)
        alpha (float): alpha

    Returns:
        float: value in t -- x(t)
    """
    stepsNumber: int = 100
    ts: list = np.linspace(t0, t, stepsNumber)
    xs: list = [np.array(x0)]
    h: float = (t - t0) / stepsNumber
    for i in range(stepsNumber):
        k1: npt.ArrayLike = func(ts[i], xs[-1])
        k2: npt.ArrayLike = func(ts[i] + h / 2 / alpha, xs[-1] + h / 2 / alpha * k1)
        xs.append(xs[-1] + h * ((1- alpha) * k1 + alpha * k2))
    return xs