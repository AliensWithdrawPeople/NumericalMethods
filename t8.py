from typing import Any, List
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import time

def f(t: npt.ArrayLike, x: npt.ArrayLike)->npt.ArrayLike:
    x: npt.ndarray = np.array(x)
    return np.array([998 * x[0] + 1998 * x[1], -999 * x[0] - 1999 * x[1]])

def ff(t: npt.ArrayLike, x: npt.ArrayLike)->npt.ArrayLike:
    x: npt.ndarray = np.array(x)
    return [998 * x[0] + 1998 * x[1], -999 * x[0] - 1999 * x[1]]

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

h = 0.0001
x = [0]
y = [np.array((1, 1))]
for i in range(1, 4):
    x.append(x[-1] + h)
    y.append(np.array(expl1(func=f, y0=y[-1], x0=x[-1], stepSize=h)) )
    # y.append(np.array(3 * np.array([2, -1]) * np.exp(-1 * i * h) - 5 * np.array([1, -1]) * np.exp(-1000 * i * h) ) )

vars = [(y[-1], x[-1], h / 2, h / 2 * f(t=x[-1], x=y[-1])), 
        (y[-1], x[-1], 5 * h / 12, h / 12 * (8 * f(x[-1], y[-1]) - f(x[-2], y[-2])) ),
        (y[-1], x[-1], 9 * h / 25, h / 24 * (19 * f(x[-2], y[-2]) - 5 * f(x[-3], y[-3]) + f(x[-4], y[-4])) ) ]

def Vars(y: List, x: List, h: float)->tuple:
    return [(y[-1], x[-1], h / 2, h / 2 * f(t=x[-1], x=y[-1]))]
    # return [(y[-1], x[-1], h / 2, h / 2 * f(t=x[-1], x=y[-1])), 
    #         (y[-1], x[-1], 5 * h / 12, h / 12 * (8 * f(x[-1], y[-1]) - f(x[-2], y[-2])) ),
    #         (y[-1], x[-1], 9 * h / 25, h / 24 * (19 * f(x[-2], y[-2]) - 5 * f(x[-3], y[-3]) + f(x[-4], y[-4])) ) ]


def explicit(func, y: npt.ArrayLike, x: npt.ArrayLike, h: float, mod: int)->npt.ArrayLike:
    if(mod == 1):
        return y[-1] + h * func(x[-1], y[-1])
    if(mod == 2):
        return y[-1] + h / 12 * (23 * func(x[-1], y[-1]) - 16 * func(x[-2], y[-2]) + 5 * func(x[-3] , y[-3]))
    if(mod == 3):
        return y[-1] + h / 24 * (55 * f(x[-1], y[-1]) - 59 * f(x[-2], y[-2]) + 37 * f(x[-3], y[-3]) - 9 * f(x[-4], y[-4]))

stepNum = 2
implicit(*vars[stepNum - 1])
print(explicit(f, y, x, h, 3))
print("true result:", 2 * np.array([2, -1]) * np.exp(-1 * 10**(-6)) + 3 * np.array([-1, 1]) * np.exp(-1000 * 10**(-6)))

def eulerImplicit(prev: npt.ArrayLike, h: float)->npt.ArrayLike:
    res = [0, 0]
    t1 = prev[0] + 1998 * h / (1 + 1999 * h) * prev[1]
    t2 = 1 - 998 * h + 1998 * 999 * h * h / (1 + 1999 * h)

    res[0] = t1 / t2
    res[1] = (prev[1] - 999 * h * res[0]) / (1 + 1999 * h)
    return res

delta1 = []
delta2 = []
y1 = [np.array([1., 1])]
x = [0]
y2 = [yy for yy in y]
h = 10**(-8)
for i in range(100):
    y1.append(explicit(f, y1, x, h, 1))
    # y2.append(explicit(f, y2, x, h, 2))
    tmp = 2 * np.array([2, -1]) * np.exp(-1 * x[-1]) + 3 * np.array([-1, 1]) * np.exp(-1000 * x[-1])
    x.append(x[-1] + h)
    delta1.append(np.linalg.norm(y1[-1] - tmp))
print(max(delta1))
# plt.plot(x[4:], delta1, color='red')
# plt.plot(x[4:], delta2)
# plt.show()

delt1 = []
delt2 = []
h = 1
steps = []
y1 = [np.array([1., 1])]
x = [0]
y2 = y
d1 = []
d2 = []

for i in range(4, 16):
    h = 10**(-i)
    x1 = [0, h, 2 * h]
    y2 = [np.array([1., 1]), 2 * np.array([2, -1]) * np.exp(-1 * h) + 3 * np.array([-1, 1]) * np.exp(-1000 * h),
            2 * np.array([2, -1]) * np.exp(-1 * 2 * h) + 3 * np.array([-1, 1]) * np.exp(-1000 * 2 * h)]
    for j in range(100):
        y1.append(explicit(f, y1, x, h, 1))
        y2.append(explicit(f, y2, x1, h=h, mod=2))
        x.append(x[-1] + h)
        x1.append(x1[-1] + h)
        tmp = 2 * np.array([2, -1]) * np.exp(-1 * x[-1]) + 3 * np.array([-1, 1]) * np.exp(-1000 * x[-1])
        tmp2 = 2 * np.array([2, -1]) * np.exp(-1 * x1[-1]) + 3 * np.array([-1, 1]) * np.exp(-1000 * x1[-1])
        d1.append(np.linalg.norm(y1[-1] - tmp))
        d2.append(np.linalg.norm(y2[-1] - tmp2))
    delt1.append(max(d1) / 100)
    delt2.append(max(d2) / 100)
    steps.append(h)
    y1 = [np.array([1., 1])]
    x = [0]
    d1 = []
    d2 = []

plt.plot(steps, delt1, color='red')
plt.plot(steps, delt2)
plt.xscale('log')
plt.yscale('log')
plt.show()