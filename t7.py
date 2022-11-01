import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from scipy.misc import derivative

a: float = 10
b: float = 2
c: float = 2
d: float = 10


def f(t: npt.ArrayLike, x: npt.ArrayLike)->npt.ArrayLike:
    x: npt.ndarray = np.array(x)
    return np.array([a * x[0] - b * x[0] * x[1], c * x[0] * x[1] - d * x[1]])


def log_dec(func):
    def wrap(*args, **kwargs):
        a = func(*args, **kwargs)
        print(a)
        return a
    return wrap
stepsize = 1 / 120

@log_dec
def Jacobian(x: npt.ArrayLike)->npt.ArrayLike:
    return np.array([[a - b * x[1], -b * x[0]], [c * x[1], -d + c * x[0]]])

def g(x: float, y: float, h: float):
    return h * (a + x - b * x * y / (1 - h * c * x + h * d))

def newtonImlicit(func, y0: npt.ArrayLike, x0: float, stepSize: float, J, constA: float, vecA: npt.ArrayLike)->npt.ArrayLike:
    prev = np.array(y0)
    # matrInv = np.linalg.inv(stepSize * J(prev))
    xPrev = y0[0]
    h = stepSize
    for i in range(1):
        x = xPrev + g(xPrev, prev[1], h)
        # x1 = (((1 - h * a) * (1 + h * d) + prev[1] + h * c * xPrev) + \
        #     np.sqrt(((1 - h * a) * (1 + h * d) + prev[1] + h * c * xPrev)**2 - 4 * (1 + h * d) * xPrev * (1 - h * a) * h * c ) )\
        #         / 2 / ((1 - h * a) * h * c)
        # x2 = (((1 - h * a) * (1 + h * d) + prev[1] + h * c * xPrev) - \
        #     np.sqrt(((1 - h * a) * (1 + h * d) + prev[1] + h * c * xPrev)**2 - 4 * (1 + h * d) * xPrev * (1 - h * a) * h * c) )\
        #         / 2 / ((1 - h * a) * h * c)
        # print(x1, ";", x2)
        # x = x2
        # x = x1 if x1 > 0 else x2
        y = prev[1] / (1 - h * c * x + h * d)
        xPrev = x
        prev = np.array([x, y])
    return prev

def implicit(y0: npt.ArrayLike, x0: float, stepSize, constA: float, vecA: npt.ArrayLike):
    h = stepSize
    return newtonImlicit(func=f, y0=y0, x0=x0, stepSize=h, J=Jacobian, constA=constA, vecA=vecA)

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
    stepsNumber: int = 10
    ts: list = np.linspace(t0, t, stepsNumber)
    xs: list = [np.array(x0)]
    h: float = (t - t0) / stepsNumber
    for i in range(stepsNumber):
        k1: npt.ArrayLike = func(ts[i], xs[-1])
        k2: npt.ArrayLike = func(ts[i] + h / 2 / alpha, xs[-1] + h / 2 / alpha * k1)
        # xs.append(xs[-1] + h * ((1- alpha) * k1 + alpha * k2))
        xs.append( implicit(xs[i], ts[i], stepSize=h, constA=h, vecA=np.array([0, 0])) )
    return xs

# for i in range(10):
#     ans: np.ndarray = RK2(f, 0, [10 + i / 10, 10 + i / 10], 1)
#     x: np.ndarray = np.array([x[0] for x in ans])
#     y: np.ndarray = np.array([y[1] for y in ans])
#     plt.plot(x, y)

ans: np.ndarray = RK2(f, 0, [5, 5], 1)
x: np.ndarray = np.array([x[0] for x in ans])
y: np.ndarray = np.array([y[1] for y in ans])
print(x)
print(y)
plt.plot(x, y)
plt.show()