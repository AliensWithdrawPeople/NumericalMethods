from math import *
import numpy as np
from scipy.misc import derivative

u: float = 1
a: float = 1
g2: float = u * a**2

def f(x)->float:
    return x*tan(x) - sqrt(g2-x**2)
    # return tan(x) - x

accuracy: float = 1e-6

#dihotomy
a: float = 0
b: float = pi / 2 - 1e-4 if g2-(pi / 2 - 1e-4)**2 >= 0 else g2**0.5 - 1e-4
iter: int = int(log(1/accuracy) / log(2))

for i in range(1, iter):
    if f((b+a)/2) * f(a) <= 0 :
        b: float = (b+a) / 2
    else:
        a: float = (b+a) / 2   

print('dichotomy val =', (b+a)/2, 'iters =', iter)


def g(x)->float:
    return sqrt(g2-x**2) / tan(x)
accuracy: float= 1e-9
#simple iterations
x:float = pi / 2 - 1e-4 if g2-(pi / 2 - 1e-4)**2 >= 0 else g2**0.5 - 1e-4
x:float = 0.7 * x
l:float= derivative(f, (b+a)/2, 1e-5, 1, order=11)
k: int = int(-log(1 / accuracy) / log( abs( derivative(lambda xx:xx - f(xx) / l, (b+a)/2.2, 1e-5, 1, order=11) ) ) )
for i in range(1, k):
    x: float = x - f(x) / l

print('iter val =', x, "iters =", k)

#newton method
x:float = pi / 2 - 1e-4 if g2-(pi / 2 - 1e-4)**2 >= 0 else g2**0.5 - 1e-4
x:float = 0.7 * x
iter: int = int(-log(1 / accuracy) / log( abs( derivative(lambda xx:xx - f(xx) / derivative(f, xx, 1e-7, 1, order=11), x, 1e-7, 1, order=11) ) ) )
# iter: int = 5s
for i in range(1, iter):
    x: float = x - f(x) / derivative(f, x, 1e-7, 1, order=11)
print('newton val =', x, "iters =", iter)

def pascalsTriangle(n: int) -> list:
    """ Calculate nth row of Pascal's triangle

    Args:
        n (int): number of row

    Returns:
        list: nth row of Pascal's triangle
    """
    if n == 0:
        return []
    elif n == 1:
        return [1]
    else:
        new: list[int] = [1]
        last: list[int] = pascalsTriangle(n - 1)
        for i in range(len(last) - 1):
            new.append(last[i] + last[i + 1])
        new += [1]
    return new

print(list(range(2)))