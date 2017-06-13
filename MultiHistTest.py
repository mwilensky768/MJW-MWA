import numpy as np
import matplotlib.pyplot as plt

A = np.random.randn(10000)
B = np.random.standard_cauchy(10000)

plt.figure(1)
plt.hist((A,B), bins = 100, range = (-5,5), histtype = 'step', label = ('FUCK','YOU'))
plt.legend()

plt.show()
