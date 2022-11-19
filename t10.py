import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

def tridiagMatrixAlg(hx: float, ht: float, N: int, d: npt.ArrayLike)->npt.ArrayLike:

    a = N * [-ht / 2 / hx / hx]
    b = N * [1 + ht / hx / hx]
    c = N * [-ht / 2 / hx / hx]

    a[0] = 0
    c[N - 1] = 0

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

L = 1
xBounds = (0, L)
tBounds = (0, 1)

N = 100
hx = np.fabs(xBounds[1] - xBounds[0]) / N
ht = np.fabs(tBounds[1] - tBounds[0]) / N
xs = np.linspace(xBounds[0], xBounds[1], N)
ts = np.linspace(tBounds[0], tBounds[1], N)

ys = [[xx * (1 - xx / L)**2 for xx in xs]]

for t in ts:
    d = N * [0]
    for i in range(1, N - 1):
        d[i] = ys[-1][i] + ht / 2 * (ys[-1][i + 1] - 2 * ys[-1][i] + ys[-1][i - 1]) / hx / hx
    ys.append(tridiagMatrixAlg(hx, ht, N, d))

for y in ys[:10]:
    plt.plot(xs, y)
plt.show()