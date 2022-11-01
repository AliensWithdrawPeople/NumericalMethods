from typing import Any, List
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

# @log_dec
def implicit(y0: npt.ArrayLike, x0: float, constA: float, vecA: npt.ArrayLike):
    J = np.array([[998, 1998], [-999, -1999]])
    return newtonImlicit(func=f, y0=y0, x0=x0, stepSize=h, J=J, constA=constA, vecA=vecA)

h = 0.001
x = [0]
y = [np.array((1, 2))]
for i in range(1, 4):
    x.append(x[-1] + h)
    y.append(np.array(expl1(func=f, y0=y[-1], x0=x[-1], stepSize=h)) )
    # y.append(np.array(3 * np.array([2, -1]) * np.exp(-1 * i * h) - 5 * np.array([1, -1]) * np.exp(-1000 * i * h) ) )

vars = [(y[-1], x[-1], h / 2, h / 2 * f(t=x[-1], x=y[-1])), 
        (y[-1], x[-1], 5 * h / 12, h / 12 * (8 * f(x[-1], y[-1]) - f(x[-2], y[-2])) ),
        (y[-1], x[-1], 9 * h / 25, h / 24 * (19 * f(x[-2], y[-2]) - 5 * f(x[-3], y[-3]) + f(x[-4], y[-4])) ) ]

def Vars(y: List, x: List)->tuple:
    return [(y[-1], x[-1], h / 2, h / 2 * f(t=x[-1], x=y[-1])), 
            (y[-1], x[-1], 5 * h / 12, h / 12 * (8 * f(x[-1], y[-1]) - f(x[-2], y[-2])) ),
            (y[-1], x[-1], 9 * h / 25, h / 24 * (19 * f(x[-2], y[-2]) - 5 * f(x[-3], y[-3]) + f(x[-4], y[-4])) ) ]


def explicit(func, y: npt.ArrayLike, x: npt.ArrayLike, h: float, mod: int)->npt.ArrayLike:
    if(mod == 1):
        return y[-1] + h * func(x[-1], y[-1])
    if(mod == 2):
        return y[-1] + h / 12 * (23 * func(x[-1], y[-1]) - 16 * func(x[-2], y[-2]) + 5 * func(x[-3] , y[-3]))
    if(mod == 3):
        return y[-1] + h / 24 * (55 * f(x[-1], y[-1]) - 59 * f(x[-2], y[-2]) + 37 * f(x[-3], y[-3]) - 9 * f(x[-4], y[-4]))

stepNum = 2
implicit(*vars[stepNum - 1])
print(explicit(f, y, x, h, stepNum))
print("true result:", 3 * np.array([2, -1]) * np.exp(-1 * 4 * h) - 5 * np.array([1, -1]) * np.exp(-1000 * 4 * h))

delta = []
u = []
v = []
uTr = []
vTr = []
for i in range(100):
    y.append(implicit(*Vars(y=y, x=x)[0]))
    x.append(x[-1] + h)
    u.append(y[-1][0])
    v.append(y[-1][1])
    tmp = 3 * np.array([2, -1]) * np.exp(-1 * x[-1]) - 5 * np.array([1, -1]) * np.exp(-1000 * x[-1])
    delta.append(np.abs(y[-1] - tmp))
    uTr.append(tmp[0])
    vTr.append(tmp[1])
# plt.plot(delta, x[4:])
# plt.plot(u, v)
plt.plot(uTr, vTr)
plt.show()