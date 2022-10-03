from functools import reduce
from scipy import integrate
from scipy.misc import derivative
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

def f(x:float)->float:
    return 1 / (1+x**2)

def g(x:float)->float:
    return x**(1/3) * np.exp(np.sin(x))

def IntTrapezoidal(func, a: float, b: float, n: int)->float:
    return (b-a) / n * (reduce(lambda acc, x: acc + func(a + x * (b-a) / n), range(1, n), 0) + (func(a) + func(b)) / 2)

def IntSimpson(func, a: float, b: float, n: int)->float:
    return (b-a) / n / 3 *(reduce(lambda acc, x: acc + (3 - (-1)**x) * func(a + x*(b-a) / n), range(1, n), func(a)) + func(b))

def IntRightRect(func, a: float, b: float, n: int)->float:
    return (b-a) / n * (reduce(lambda acc, x: acc + func(a + (x + 1) * (b-a) / n), range(0, n), 0))
    
def IntAvg(func, a: float, b: float, n: int)->float:
    return (b-a) / n * (reduce(lambda acc, x: acc + func((a + x * (b-a) / n + (x + 1) * (b-a) / n) / 2), range(0, n), 0))

def IntRunge(func, method, a: float, b: float, n: int, rho: int, r: float, notErr: bool)->float:
    """Runge method for improving numerical integration 

    Args:
        func (_type_): _description_
        method (_type_): numerucal integration method
        a (float): lower boundary
        b (float): higher boundary
        n (int): number of points
        rho (int): accuracy order
        r (float): some number
        notErr (bool): are you evaluating error of method or not?

    Returns:
        float: integral
    """
    return (method(func, a, b, n) * (r**rho if notErr else 1) - method(func, a, b, int(n / r)) / (r**rho - 1))

a: float = 0
b: float = 1
n: int = 16
print("Trapezoidal rule I = ", IntTrapezoidal(g, a, b, n))
print("Simpson rule I = ", IntSimpson(g, a, b, n))
print("Simpson rule I = ", IntRunge(g, IntSimpson, a, b, n, 4, 0.1, True))
print("Right rectangle rule I = ", IntRightRect(g, a, b, n))
print("Average rule I = ", IntAvg(g, a, b, n))
print("SciPy I = ", integrate.quad(g, a, b, epsabs = 1e-10, epsrel=1e-6, limit=5000)[0])

errTrapezoidal: list = []
errSimpson: list = []
errRightRect: list = []
errAvg: list = []
errRunge: list = []

ns: list = [2**i for i in range(2, 10)]
for n in  ns:
    h: float = (b - a) / n
    errTrapezoidal.append(h**2 / 12 * IntTrapezoidal(lambda x: derivative(g, x, dx=1e-4, n=2, order=7), a, b, n))
    errSimpson.append(-h**4 / 180 * IntSimpson(lambda x: derivative(g, x, dx=1e-4, n=4, order=7), a, b, n))
    errRunge.append(IntRunge(g, IntSimpson, a, b, n, 4, 1.5, False))
    errRightRect.append( -h / 2 * IntRightRect(lambda x: derivative(g, x, dx=1e-4, n=1, order=7), a, b, n))
    errAvg.append(-h**2 / 24 * IntAvg(lambda x: derivative(g, x, dx=1e-3, n=2, order=7), a, b, n))
# plt.plot(ns, errTrapezoidal, c="red")
# plt.plot(ns, errAvg, c="green")
# plt.plot(ns, errSimpson, c="blue")
plt.plot(ns, errRunge, c="blue")
plt.show()