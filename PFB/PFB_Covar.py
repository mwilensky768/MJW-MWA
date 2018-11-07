import numpy as np
from SSINS import plot_lib
import matplotlib.pyplot as plt

n = 512
M = 128
NT1 = 8
NT2 = 12
w_k = 2 * np.pi * np.arange(n) / n
w_l = 2 * np.pi * np.arange(M) / (M * n)
beta = 5
W_2_coeff = '/Users/mikewilensky/Repos/MJW-MWA/Useful_Information/pfb2coeff.csv'
W_1 = np.kaiser(n * NT1, beta)
W_2 = np.genfromtxt(W_2_coeff, delimiter=',')
W_2 /= W_2.max()
t = np.arange(n * M * NT1 * NT2)

w = 2 * np.pi * np.arange(n * M) / (n * M)
var = np.arange(n * M)
var[0] = var[1]
var = var**(-2.2)
covar_freq = np.diag(var)
FT = np.matrix(np.exp(1.0j * np.outer(w, np.arange(n * M))))
covar_time = np.matmul(FT.H, covar_freq)
covar_time = np.matmul(covar_time, FT) / (n * M)
fig, ax = plt.subplots(figsize=(14, 8))
plot_lib.image_plot(fig, ax, covar_time, ylabel='Time Index', xlabel='Time Index',
                    cbar_label='Variance')
fig.savefig('/Users/mikewilensky/PFB/Covar_Plot.png')
