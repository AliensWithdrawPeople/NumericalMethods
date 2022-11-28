import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from functools import reduce

def DFT(x)->npt.ArrayLike:
    x1 = np.asarray(x, dtype=float)
    N = x1.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    return np.dot(x1, np.exp(-2j * np.pi * k * n / N))

def hanning(x):
    x1 = np.asarray(x, dtype=float)
    N = x1.shape[0]
    k = np.arange(N)
    return x1 * 0.5 * (1 - np.cos(2 * np.pi * k / N))

N = 1000
bounds = (0, 2 * np.pi)
xs = np.linspace(*bounds, N)
yF = np.sin(5.1 * xs) + 0.002 * np.sin(25.5 * xs)
yF1 = hanning(yF)
yF1 = DFT(yF1)
yF = DFT(yF)
plt.plot(np.absolute(yF))
plt.plot(np.absolute(yF1), color='red')
plt.show()