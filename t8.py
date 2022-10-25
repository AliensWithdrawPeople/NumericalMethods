from typing import Any
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import time

def f(t: npt.ArrayLike, x: npt.ArrayLike)->npt.ArrayLike:
    x: npt.ndarray = np.array(x)
    return np.array([998 * x[0] + 1998 * x[1], -999 * x[0] - 1999 * x[1]])

def newtonImlicit(func, y0: npt.ArrayLike, x0: float, stepSize: float, J: npt.ArrayLike, constA: float, vecA: npt.ArrayLike)->npt.ArrayLike:
    matrInv = np.linalg.inv(np.identity(2) - stepSize * J)
    prev = np.array(y0)
    for i in range(10):
        res = prev + vecA + matrInv.dot(y0 - prev + constA * np.array(func(t=x0, x=prev)))
        prev = res
    return res

def expl1(func, y0: npt.ArrayLike, x0: float, stepSize: float)->npt.ArrayLike:
    return y0 + stepSize * func(x0, y0)

def impl1(y0: npt.ArrayLike, x0: float, stepSize: float = 0.001)->npt.ArrayLike:
    J = np.array([[998, 1998], [-999, -1999]])
    return newtonImlicit(func=f,
                        y0=y0,
                        x0=x0,
                        stepSize=stepSize,
                        J=J,
                        constA=stepSize / 2,
                        vecA=stepSize / 2 * f(t=x0, x=y0))

def log_dec(func):
    def wrap(*args, **kwargs):
        begin = time.time()
        a = func(*args, **kwargs)
        end = time.time()
        print(a, "and this shit (", func.__name__, ") took", end - begin, "seconds of your life.")
        return a
    return wrap

@log_dec
def implicit(y0: npt.ArrayLike, x0: float, constA: float, vecA: npt.ArrayLike):
    J = np.array([[998, 1998], [-999, -1999]])
    return newtonImlicit(func=f, y0=y0, x0=x0, stepSize=h, J=J, constA=constA, vecA=vecA)

h = 0.0001
x = [0]
y = [np.array((1, 2))]
for i in range(1, 4):
    x.append(x[-1] + h)
    y.append(np.array(expl1(func=f, y0=y[-1], x0=x[-1], stepSize=h)) )
    # y.append(np.array(3 * np.array([2, -1]) * np.exp(-1 * i * h) - 5 * np.array([1, -1]) * np.exp(-1000 * i * h) ) )

vars = [(y[0], x[0], h / 2, h / 2 * f(t=x[0], x=y[0])), 
        (y[1], x[2], 5 * h / 12, h / 12 * (8 * f(x[1], y[1]) - f(x[0], y[0])) ),
        (y[2], x[2], 9 * h / 25, h / 24 * (19 * f(x[2], y[2]) - 5 * f(x[1], y[1]) + f(x[0], y[0])) ) ]

def explicit(func, y: npt.ArrayLike, x: npt.ArrayLike, h: float, mod: int)->npt.ArrayLike:
    if(mod == 1):
        return y[0] + h * func(x[0], y[0])
    if(mod == 2):
        return y[2] + h / 12 * (23 * func(x[2], y[2]) - 16 * func(x[1], y[1]) + 5 * func(x[0] , y[0]))
    if(mod == 3):
        return y[3] + h / 24 * (55 * f(x[3], y[3]) - 59 * f(x[2], y[2]) + 37 * f(x[1], y[1]) - 9 * f(x[0], y[0]))

stepNum = 2
implicit(*vars[stepNum - 1])
print(explicit(f, y, x, h, stepNum))
print(y[stepNum])
print("true result:", 3 * np.array([2, -1]) * np.exp(-1 * stepNum * h) - 5 * np.array([1, -1]) * np.exp(-1000 * stepNum * h))

def wrapper():
    ys = y
    xs = np.linspace(0, 1, h)
    