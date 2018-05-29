import numpy as np


def emp_cdf(n, x):
    A = np.random.rayleigh(scale=1. / np.sqrt(2 * np.log(2)), size=(n, 1e9)).mean(axis=0)
    return(np.count_nonzero(A < x) / 1e9)


for x in np.linspace(0, max(A), num=1e9):
    cdf = emp_cdf(2, x)
