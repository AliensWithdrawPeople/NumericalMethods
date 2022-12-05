import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from collections import namedtuple

conductivityX = 1
conductivityY = 0.5
L = 1
N = 100
xBounds = (0, L)
yBounds = (0, L)
tBounds = (0, 1)
Slice = namedtuple('Slice', ['x', 'y'])


def tma(hx: float, ht: float, N: int, d: npt.ArrayLike)->npt.ArrayLike:
    """Tridiagonal matrix algorithm
    Args:
        hx (float): spatial step
        ht (float): time step
        N (int): grid size
        d (npt.ArrayLike): right part 

    Returns:
        npt.ArrayLike: solution of one-dimensional heat equation at t = t0 + ht 
        as a grid function with spatial step = hx.
    """
    a = N * [-ht / 2 / hx / hx]
    b = N * [1 + ht / hx / hx]
    c = N * [-ht / 2 / hx / hx]

    a[0] = 0
    b[0] = 1
    c[0] = 0

    c[N - 1] = 0
    b[N - 1] = 1
    a[N - 1] = 0

    for i in range(1, N):
        ksi = a[i] / b[i-1]
        a[i] = 0
        b[i] = b[i] - ksi * c[i - 1]
        d[i] = d[i] - ksi * d[i - 1]

    ys = N * [0]
    ys[N - 1] = d[N-1] / b[N - 1]
    i = N - 2
    while i > -1:
        ys[i] = 1 / b[i] * (d[i] - c[i] * ys[i + 1])
        i = i - 1
    return ys

def solution_step(
        N: int, spaceStep: float, timeStep: float, mode: int, 
        prevSlice: Slice, boundary_conditions: tuple)->Slice:
    hx = spaceStep
    ht = timeStep
    prev = prevSlice[mode]
    d = N * [0]
    for i in range(1, N - 1):
        d[i] = prev[i] + ht / 2 * (prev[i + 1] - 2 * prev[i] + prev[i - 1]) / hx / hx
    d[0], d[N - 1] = boundary_conditions[mode]
    if (mode == 0):
        return Slice(tma(hx, ht, N, d), prevSlice[1])
    else:
        print(d[N - 1])
        return Slice(prevSlice[0], tma(hx, ht, N, d)) 

xBounds_ = (conductivityX**0.5 * xBounds[0], conductivityX**0.5 * xBounds[1])
yBounds_ = (conductivityY**0.5 * yBounds[0], conductivityY**0.5 * yBounds[1])
hx = np.fabs(xBounds_[1] - xBounds_[0]) / N
hy = np.fabs(yBounds_[1] - yBounds_[0]) / N
ht = np.fabs(tBounds[1] - tBounds[0]) / N
xs = np.linspace(xBounds_[0], xBounds_[1], N)
ys = np.linspace(yBounds_[0], yBounds_[1], N)
ts = np.linspace(tBounds[0], tBounds[1], 2 * N)

us = [Slice([np.sin(np.pi * x) for x in xs], [np.sin(np.pi * y) for y in ys])]
boundary_conditions = [zip(np.sin(2 * np.pi * ts), np.sin(2 * np.pi * ts)), zip(2 * N * [0], 2 * N * [0])]
boundary_conditions = list(zip(zip(np.sin(2 * np.pi * ts), np.sin(2 * np.pi * ts)), zip(2 * N * [0], 2 * N * [0])))
for i in range(1, N - 1):
    tmp = solution_step(N, hx, ht / 2, 0, us[-1], boundary_conditions[2 * i])
    us.append(solution_step(N, hy, ht / 2, 1, tmp, boundary_conditions[2 * i + 1]))

for elem in us[:1]:
    plt.plot(elem[1])
plt.show()