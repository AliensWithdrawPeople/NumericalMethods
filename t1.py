import numpy as np

for mantissa in range(1, 100):
    eps: np.float32 = 2**(-mantissa)
    # if np.float32(1 + eps / 2)  == 1:
    if (1 + eps / 2)  == 1:
        print(np.log2(eps))
        print(mantissa)
        break

for i in range (50, 10000):
    if np.float32(2**(-i))  == 0:
    # if 2**(1/i) == 1:
        print(i-1)
        break
