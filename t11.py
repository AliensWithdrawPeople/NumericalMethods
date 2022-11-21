import io
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

# def tridiagMatrixAlg(N: int, a: npt.ArrayLike, b: npt.ArrayLike, c: npt.ArrayLike, d: npt.ArrayLike)->npt.ArrayLike:
def tridiagMatrixAlg(N: int, *args)->npt.ArrayLike:
    a, b, c, d = [arg.copy() for arg in args]
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
    return np.array(ys)

boundaries = (-10, 10)
N = 5000
eps = 10**(-7)
h = np.fabs(boundaries[1] - boundaries[0]) / N
xs = np.linspace(*boundaries, N)
psiPrev = np.exp(-xs**2 / 2)
psi0 = np.exp(-xs**2 / 2)

a = N * [-0.5 / h / h]
b = np.array(N * [1 / h / h]) + xs**2 / 2
c = N * [-0.5 / h / h]
a[0] = 0
c[N - 1] = 0

energy = 0
energyPrev = 1
while np.fabs(energy - energyPrev) > eps:
    energyPrev = energy
    psi = tridiagMatrixAlg(N, a, b - energy, c, psiPrev)
    energy = energy + np.linalg.norm(psiPrev) / np.linalg.norm(psi)
    psiPrev = psi[:]
print(energy)