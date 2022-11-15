import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


def tridiagMatrixAlg(L1: float, L2: float, N: int)->npt.ArrayLike:
    h = (L1 - L2) / N
    x = np.linspace(L1, L2, N)
    d = np.cos(x)

    a = N * [1 / h / h]
    b = N * [-2 / h / h]
    c = N * [1 / h / h]

    a[0] = 0
    b[0] = 1
    c[0] = 0
    d[0] = 0

    a[N - 1] = 1/h
    b[N - 1] = -1/h
    c[N - 1] = 0
    d[N - 1] = 2

    for i in range(1, N):
        ksi = a[i] / b[i-1]
        a[i] = 0
        b[i] = b[i] - ksi * c[i - 1]
        d[i] = d[i] - ksi * d[i - 1]

    ys = N * [0]
    ys[N - 1] = d[N-1] / b[N - 1]
    i = N - 2
    while i > 1:
        ys[i] = 1 / b[i] * (d[i] - c[i] * ys[i + 1])
        i = i - 1
    return ys

L1 = -np.pi / 2
L2 = np.pi / 2
x = np.linspace(L1, L2, 1000)
ys = tridiagMatrixAlg(L1, L2, 1000)
yTr = [-np.cos(xx) + 1 * xx + np.pi / 2 for xx in x]
plt.plot(x, ys, color='blue')
plt.plot(x, yTr, color='red')
plt.show()

# delta = []
# steps = [10**i for i in range(2, 6)]
# for N in steps:
#     x = np.linspace(L1, L2, N)
#     ys = tridiagMatrixAlg(L1, L2, N)
#     yTr = [-np.cos(xx) + 1 * xx + np.pi / 2 for xx in x]
#     delta.append(max( list(map(lambda yy: np.fabs(yy) , np.array(ys) - np.array(yTr) ) ) ) )

# plt.plot(steps, delta, color='blue')
# plt.xscale('log')
# plt.yscale('log')
# plt.show()