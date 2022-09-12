import numpy as np

def f(t: float, x: float)->float:
    return -x

def EulerMethod(func, t0: float, x0: float, t: float)->float:
    """ First-order numerical procedure for solving ordinary differential equations (dx/dt = f(t, x))
    with a given initial value x(t0) = x0

    Args:
        func (Any): function f in dx/dt = f(t, x)
        t0 (float): point of initial value
        x0 (float): initial value x(t0) = x0
        t (float): point where to evaluate solution x(t)

    Returns:
        float: value in t -- x(t)
    """
    stepsNumber: int = 10000
    ts: list = np.linspace(t0, t, stepsNumber)
    xs: list = [x0]
    h: float = (t - t0) / stepsNumber
    for i in range(stepsNumber):
        xs.append(xs[-1] + h * func(ts[i], xs[-1]))
    return xs[-1]

def RK2(func, t0: float, x0: float, t: float, alpha: float = 3 / 4)->float:
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

def RK4(func, t0: float, x0: float, t: float)->float:
    """ The fourth order Runge-Kutta method

    Args:
        func (Any): function f in dx/dt = f(t, x)
        t0 (float): point of initial value
        x0 (float): initial value x(t0) = x0
        t (float): point where to evaluate solution x(t)

    Returns:
        float: value in t -- x(t)
    """
    stepsNumber: int = 10000
    ts: list = np.linspace(t0, t, stepsNumber)
    xs: list = [x0]
    h: float = (t - t0) / stepsNumber
    for i in range(stepsNumber):
        k1: float = func(ts[i], xs[-1])
        k2: float = func(ts[i] + h / 2, xs[-1] + h / 2 * k1)
        k3: float = func(ts[i] + h / 2, xs[-1] + h / 2 * k2)
        k4: float = func(ts[i] + h, xs[-1] + h * k3)
        xs.append(xs[-1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    return xs[-1]

ts: list = np.linspace(0, 3, 100)
# print(max([np.fabs(EulerMethod(f, 0, 1, t) - np.exp(-t)) for t in ts]))
# print(max([np.fabs(RK2(f, 0, 1, t) - np.exp(-t)) for t in ts]))
print(max([np.fabs(RK4(f, 0, 1, t) - np.exp(-t)) for t in ts]))