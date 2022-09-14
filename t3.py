from functools import reduce
from scipy import integrate
from scipy.misc import derivative
from scipy.optimize import minimize
import numpy as np

def f(x:float)->float:
    return 1 / (1+x**2)

def g(x:float)->float:
    return x**(1/3) * np.exp(np.sin(x))

def IntTrapezoidal(func, a: float, b: float, n: int)->float:
    return (b-a) / n * (reduce(lambda acc, x: acc + func(a + x * (b-a) / n), range(1, n), 0) + (func(a) + func(b)) / 2)

def IntSimpson(func, a: float, b: float, n: int)->float:
    val: float = (b-a) / n / 3 *(reduce(lambda acc, x: acc + (3 - (-1)**x) * func(a + x*(b-a) / n), range(1, n), func(a)) + func(b))
    return val

a: float = 0
b: float = 1
n: int = 5000
print(IntTrapezoidal(g, a, b, n))
print(IntSimpson(g, a, b, n))
print(integrate.quad(g, a, b, epsabs = 1e-10, epsrel=1e-6, limit=5000)[0])

r = [1, 2, 3]
print(r[-2])