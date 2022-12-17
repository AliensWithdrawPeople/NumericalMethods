import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

conductivityX = 1
conductivityY = 0.5
L = 1
N = 101
Nt = 101
xBounds = (0, 2 * L)
yBounds = (0, 2 * L)
tBounds = (0, 2)


def tma(hx: float, ht: float, N: int, d: npt.ArrayLike, conductivity: float)->npt.ArrayLike:
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
    a = N * [-conductivity * ht / (2 * hx**2)]
    b = N * [1 + conductivity * ht / hx / hx]
    c = N * [-conductivity * ht / (2 * hx**2)]

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

xBounds_ = (xBounds[0], xBounds[1])
yBounds_ = (yBounds[0], yBounds[1])
hx = np.fabs(xBounds_[1] - xBounds_[0]) / N
hy = np.fabs(yBounds_[1] - yBounds_[0]) / N
ht_ = np.fabs(tBounds[1] - tBounds[0]) / Nt
xs = np.linspace(xBounds_[0], xBounds_[1], N)
ys = np.linspace(yBounds_[0], yBounds_[1], N)
ts = np.linspace(tBounds[0], tBounds[1], 2 * N)
ts1 = np.linspace(tBounds[0], tBounds[1], N)
print(hx, hy, ht_)

def solution_step(
        N: int, xStep: float, yStep: float, timeStep: float, 
        prev: npt.ArrayLike, t: float)->npt.ArrayLike:
    hx = xStep
    hy = yStep
    ht = timeStep
    d = N * [0]

    tmp = [N * [np.sin(2 * np.pi * t)]]
    for k in range(1, N - 1):
        for i in range(N - 1):
            d[i]= prev[i][k] + (ht/(4 * hy * hy))*(prev[i][k+1]- 2 * prev[i][k] + prev[i][k-1])
        d[0] = np.sin(2 * np.pi * t)
        d[N - 1] = np.sin(2 * np.pi * t)
        tmp.append(tma(hx, ht, N, d, conductivityX))
    tmp.append(N * [np.sin(2 * np.pi * t)])

    res = [N * [0]]
    for k in range(1, N - 1):
        for i in range(N - 1):
            d[i] = tmp[k][i] + (ht / (2 * hx * hx))*(tmp[k][i+1]- 2 * tmp[k][i] + tmp[k][i-1])
        d[0] = 0
        d[N - 1] = 0
        res.append(tma(hx, ht, N, d, conductivityY))
    res.append(N * [0])

    return np.array(res)

us = [np.sin(np.pi * xs).reshape((N, 1)) * (np.sin(np.pi * ys))]
# us = [np.zeros((N, 1)) * np.zeros(N)]

for i in range(N):
    # tmp = solution_step(N, hx, ht / 2, True, us[-1], ts[2 * i + 1])
    us.append(solution_step(N, hx, hy, ht_, us[-1], ht_ * i))

maxes = [elem[50][50] for elem in us]
true_maxes = np.exp(-np.pi**2 * ts) * np.sin(2 * np.pi * ts)
# true_maxes = np.exp(-3 / 2 * np.pi**2 * ts)
plt.plot(maxes)
plt.plot(true_maxes[:N])

Xgraph, Ygraph = np.meshgrid(xs, ys)
fig = plt.figure(figsize=(10,7))
ax1 = fig.add_subplot(221, projection='3d')
ax2 = fig.add_subplot(222, projection='3d')
ax3 = fig.add_subplot(212)

ax1.plot_surface(Xgraph, Ygraph, np.transpose(us[0]), cmap='coolwarm')

plt.show()