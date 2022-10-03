from functools import reduce
from scipy import misc
from scipy.special import jv
import numpy as np
import matplotlib.pyplot as plt

def IntTrapezoidal(func, a: float, b: float, n: int)->float:
    return (b-a) / n * (reduce(lambda acc, x: acc + func(a + x * (b-a) / n), range(1, n), 0) + (func(a) + func(b)) / 2)

def IntSimpson(func, a: float, b: float, n: int)->float:
    return (b-a) / n / 3 *(reduce(lambda acc, x: acc + (3 - (-1)**x) * func(a + x*(b-a) / n), range(1, n), func(a)) + func(b))

def IntRunge(func, method, a: float, b: float, n: int, rho: int, r: float, notErr: bool)->float:
    return (method(func, a, b, n) * (r**rho if notErr else 1) - method(func, a, b, int(n / r)) / (r**rho - 1))

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

# x: float = 1
m: int = 1
n: int = 100
dx: float = 1e-5
isSimps: bool = False
a = [derivative(lambda xx: bes(xx, 0, isSimps, n), x, dx, 1) + bes(x, 1, isSimps, n) for x in list(np.linspace(0, 3 * np.pi, 1500))]
print( max(a) )
xs: list = list(np.linspace(0, 3 * np.pi, 1500))
plt.plot(xs, a, c="blue")
plt.show()