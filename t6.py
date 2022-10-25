from math import fabs
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

def f(t: float, x: float)->float:
    return -x

def EulerMethod(func, t0: float, x0: float, t: float, stepsNumber: int)->float:
    """ First-order numerical procedure for solving ordinary differential equations (dx/dt = f(t, x))
    with a given initial value x(t0) = x0

    Args:
        func (Any): function f in dx/dt = f(t, x)
        t0 (float): point of initial value
        x0 (float): initial value x(t0) = x0
        t (float): point where to evaluate solution x(t)
        stepsNumber (int): number of steps

    Returns:
        float: value in t -- x(t)
    """
    ts: list = np.linspace(t0, t, stepsNumber)
    xs: list = [x0]
    h: float = (t - t0) / stepsNumber
    for i in range(stepsNumber):
        xs.append(xs[-1] + h * func(ts[i], xs[-1]))
    return xs[-1]


def RK2(func, t0: float, x0: npt.ArrayLike, t: float, stepsNumber: int, alpha: float = 3 / 4)->npt.ArrayLike:
    """ The second order Runge-Kutta method

    Args:
        func (Any): function f in dx/dt = f(t, x)
        t0 (float): point of initial value
        x0 (float): initial value x(t0) = x0
        t (float): point where to evaluate solution x(t)
        stepsNumber (int): number of steps
        alpha (float): alpha

    Returns:
        float: value in t -- x(t)
    """
    ts: list = np.linspace(t0, t, stepsNumber)
    xs: list = [np.array(x0)]
    h: float = (t - t0) / stepsNumber
    for i in range(stepsNumber):
        k1: npt.ArrayLike = func(ts[i], xs[-1])
        k2: npt.ArrayLike = func(ts[i] + h / 2 / alpha, xs[-1] + h / 2 / alpha * k1)
        xs.append(xs[-1] + h * ((1- alpha) * k1 + alpha * k2))
    return xs[-1]

def RK4(func, t0: float, x0: npt.ArrayLike, t: float)->npt.ArrayLike:
    """ The fourth order Runge-Kutta method

    Args:
        func (Any): function f in dx/dt = f(t, x)
        t0 (float): point of initial value
        x0 (float): initial value x(t0) = x0
        t (float): point where to evaluate solution x(t)

    Returns:
        float: value in t -- x(t)
    """
    stepsNumber: int = 1
    ts: list = np.linspace(t0, t, stepsNumber)
    xs: list = [np.array(x0)]
    h: float = (t - t0) / stepsNumber
    for i in range(stepsNumber):
        k1: float = func(ts[i], xs[-1])
        k2: float = func(ts[i] + h / 2, xs[-1] + h / 2 * k1)
        k3: float = func(ts[i] + h / 2, xs[-1] + h / 2 * k2)
        k4: float = func(ts[i] + h, xs[-1] + h * k3)
        xs.append(xs[-1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4))
    return xs[-1]

def StepSize(method, func, t0:float, x0:float, step: float, accuracyOrder: int, delta: float = 0.01, eps: float = 0.001)->float:
    """_summary_

    Args:
        method (_type_): _description_
        func (_type_): _description_
        t0 (float): _description_
        x0 (float): _description_
        step (float): _description_
        accuracyOrder (int): _description_
        delta (float): absolute error of this step
        eps (float): relative error of this step

    Returns:
        float: step
    """
    while True:
        y1 = method(func, t0, x0, t0 + step, 2) 
        y2 = method(func, t0, x0, t0 + 2 * step, 1)
        R = (delta + eps * fabs(y2) / 2)
        R1 = (y2 - y1) / (2**(accuracyOrder - 1) - 1)
        if(fabs(R1) / R < 1):
            # print("dd=", (fabs(R1) / R)**(-1 / accuracyOrder))
            return step * (fabs(R1) / R)**(-1 / accuracyOrder)
        step = step / 2

ts: list = list(np.linspace(0, 3, 10))
steps: list = [2**i for i in range(2, 10)]
# eu = [ np.fabs(EulerMethod(f, 0, 1, 3, stepsNumber) - np.exp(-3)) for stepsNumber in steps]
# rk2 = [ np.fabs(RK2(f, 0, 1, 3, stepsNumber) - np.exp(-3)) for stepsNumber in steps]
# rr = StepSize(RK2, f, 0, 1, 2, 2, delta=0.01, eps=0.1)
# print("step =", rr)
# print(np.fabs(RK2(f, 0, 1, 2, int(1 / rr)) - np.exp(-2)))
res = [1]
rk4 = [1]
stepSize = 1
for i in range(1, 10):
    t = ts[i]
    stepSize = StepSize(method=RK2, func=f, t0=ts[i-1], x0=res[-1], step=stepSize, accuracyOrder=2, delta=0.01, eps=0.1)
    res.append( RK2(func=f, t0=ts[i-1], x0=res[-1], t=t, stepsNumber= int( (t - ts[i-1]) / stepSize ) ))
    rk4.append( RK4(func=f, t0=ts[i-1], x0=res[-1], t=t))
    print(int( (t - ts[i-1]) / stepSize ), " : ", res[-1] - np.exp(-t))

ex = [np.exp(-t) for t in ts]
plt.plot(ts, res, c="blue")
plt.plot(ts, rk4, c="red")
# plt.plot(ts, ex, c="green")
plt.show()