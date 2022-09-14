from math import *
import numpy as np
from scipy.misc import derivative

u: float = 1
a: float = 1
g2: float = u * a**2

def f(x)->float:
    return x*tan(x) - sqrt(g2-x**2)
    # return tan(x) - x

accuracy1: float = 1e-6

#dihotomy
a: float = 0
b: float = pi / 2 - 1e-6 if g2-(pi / 2 - 1e-6)**2 >= 0 else g2**0.5 - 1e-6
x0: float = (b + a) / 2
iter: int = int(log((a + b) / 2 / accuracy1) / log(2))
count: int = 0
while 1:
    prevA: float = a
    prevB: float = b
    count: int = count + 1
    if f((b+a)/2) * f(a) <= 0 :
        b: float = (b+a) / 2
    else:
        a: float = (b+a) / 2
    if(np.fabs(f((b+a)/2) - f((prevA + prevB) / 2)) < accuracy1):
        break

print('dichotomy val =', (b+a)/2, 'iters =', count, "estim iter =", iter)


accuracy: float= 1e-8
#simple iterations
x:float = (b + a) / 2
x:float = x0
l:float = derivative(f, x, 1e-10, 1, order=11)
print((b + a) / 2)
# l:float = 3.3
count: int = 0
while 1:
    count: int = count + 1
    prevX: float = x
    x: float = x - f(x) / l
    if(np.fabs(f(x) - f(prevX)) < accuracy):
        break

iter: int = int(-log(x0 / accuracy) / log( abs( derivative(lambda xx:xx - f(xx) / l, x, 1e-10, 1, order=11) ) ) )
print(derivative(lambda xx:xx - f(xx) / l, x, 1e-10, 1, order=11))

print('iter val =', x, "iters =", count, "estim iter =", iter )

#newton method
# x:float = (a + b) / 2
x:list = [x0, x0 + 1e-4]
count: int = 0
while 1:
    count: int = count + 1
    # x: float = x - f(x) / derivative(f, x, 1e-10, 1, order=15)
    x.append(x[-1] - f(x[-1]) * (x[-1] - x[-2]) / (f(x[-1]) - f(x[-2])) )
    if(np.fabs(f(x[-1]) - f(x[-2])) < accuracy):
        break
print('newton val =', x[-1], "iters =", count)