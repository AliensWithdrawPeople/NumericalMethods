from functools import reduce
from scipy import integrate
from scipy import misc
from scipy.optimize import minimize
from scipy.special import jv
import numpy as np
import matplotlib.pyplot as plt

def IntTrapezoidal(func, a: float, b: float, n: int)->float:
    return (b-a) / n * (reduce(lambda acc, x: acc + func(a + x * (b-a) / n), range(1, n), 0) + (func(a) + func(b)) / 2)

def IntSimpson(func, a: float, b: float, n: int)->float:
    return (b-a) / n / 3 *(reduce(lambda acc, x: acc + (3 - (-1)**x) * func(a + x*(b-a) / n), range(1, n), func(a)) + func(b))

def pascalsTriangle(n: int) -> list:
    """ Calculate nth row of Pascal's triangle

    Args:
        n (int): number of row

    Returns:
        list: nth row of Pascal's triangle
    """
    if n == 0:
        return [1]
    else:
        new: list[int] = [1]
        last: list[int] = pascalsTriangle(n - 1)
        for i in range(len(last) - 1):
            new.append(last[i] + last[i + 1])
        new += [1]
    return new

def derivative(func, x: float, dx: float, n: int)->float:
    """ Calculate nth derivative of function in x using central finite differences

    Args:
        func (Any): function
        x (float): point where derivative is calculated
        dx (float): size of step in finite difference
        n (int): order of derivative

    Returns:
        float: nth derivative of func in x
    """
    binomial: list = pascalsTriangle(n)
    return reduce(lambda acc, i: acc + (-1)**i * binomial[i] * func(x + (n / 2 - i) * dx), range(n + 1), 0) / (dx**n)
   
def bes(x: float, m: int, mod: bool, n: int)->float:
    """Bessel functions of the first kind J_m with integer m

    Args:
        x (float): point where the Bessel function is calculated
        m (int): number of Bessel function
        mod (bool): True - Trapezoidal rule, False - Simpson rule
        n (int): number of points for integral

    Returns:
        float: value of J_m(x)
    """
    if mod:
        return IntTrapezoidal(lambda t: np.cos(m * t - x * np.sin(t)), 0, np.pi, n) / np.pi
    return IntSimpson(lambda t: np.cos(m * t - x * np.sin(t)), 0, np.pi, n) / np.pi

x: float = 1
m: int = 1
n: int = 2
dx: float = 5e-1
simps: bool = False
print(bes(x, m, simps, n))
print(jv(m,x))
a = [np.abs( derivative(lambda xx: bes(xx, 0, simps, n), x, dx, 1) + bes(x, 1, simps, n) ) for x in list(np.linspace(0, 2 * np.pi, 1000))]
# print( max( [np.abs( misc.derivative(lambda xx: jv(0, xx), x, dx, 1) + jv(1,x) ) for x in list(np.linspace(0, 2 * np.pi, 1000))] ) )
print( max(a) )
xs: list = list(np.linspace(0, 2 * np.pi, 1000))
plt.plot(xs, a, c="blue")
plt.show()