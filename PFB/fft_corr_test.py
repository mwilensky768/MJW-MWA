import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft


def corr(x, y):
    cor = np.sum(x * y, axis=0)
    return(cor)


x_t = np.random.rayleigh(size=[int(1e4), int(1e4)])
x_f = fft(x_t)
cor = corr(x_f.real, x_f.imag) / np.sqrt(corr(x_f.real, x_f.real) * corr(x_f.imag, x_f.imag))
cor = cor[np.logical_not(np.isnan(cor))]
var = np.var(cor)
mu = np.mean(cor)
print(var)
counts, bins, _ = plt.hist(cor, bins='auto', histtype='step')
gauss_data = np.random.normal(scale=np.sqrt(var), loc=mu, size=int(np.sum(counts)))
plt.hist(gauss_data, bins=bins, histtype='step')
plt.yscale('log')
plt.savefig('/Users/mike_e_dubs/MWA/PFB/real_imag_corr.png')
plt.close()

X_t = np.random.rayleigh(size=[int(1e2), int(1e6)])
cor = fft(X_t**2)
plt.plot(np.mean(cor * np.conj(cor), axis=0))
plt.yscale('symlog')
plt.savefig('/Users/mike_e_dubs/MWA/PFB/1-d_corr.png')
