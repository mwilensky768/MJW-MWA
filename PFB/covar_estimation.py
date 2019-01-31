import numpy as np
from scipy.fftpack import fft
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm

outpath = '/Users/mike_e_dubs/MWA/PFB'
array_path = '%s/R_f_split_norm_.npy' % outpath

if not os.path.exists(array_path):
    sz = [256, int(1e4)]
    X_t_exp = np.random.exponential(size=sz)
    X_t_norm = np.random.normal(size=sz)
    X_f_exp = fft(X_t_exp, axis=0)
    X_f_norm = fft(X_t_norm, axis=0)

    R_f_exp = np.corrcoef(X_f_exp)
    R_f_norm = np.corrcoef(X_f_norm)

    R_f_split_exp = np.corrcoef(np.concatenate([X_f_exp.real, X_f_exp.imag]))
    R_f_split_norm = np.corrcoef(np.concatenate([X_f_norm.real, X_f_norm.imag]))

    np.save('%s/R_f_exp.npy' % outpath, R_f_exp)
    np.save('%s/R_f_norm.npy' % outpath, R_f_norm)
    np.save('%s/R_f_split_exp.npy' % outpath, R_f_split_exp)
    np.save(array_path, R_f_split_norm)
else:
    R_f_exp = np.load('%s/R_f_exp.npy' % outpath)
    R_f_norm = np.load('%s/R_f_norm.npy' % outpath)
    R_f_split_exp = np.load('%s/R_f_split_exp.npy' % outpath)
    R_f_split_norm = np.load(array_path)

plt.figure(figsize=(16, 9))
plt.imshow(R_f_split_exp, norm=colors.SymLogNorm(1e-8), cmap=cm.coolwarm)
plt.colorbar()
plt.savefig('%s/R_f_split_exp.png' % outpath)
plt.close()

plt.figure(figsize=(16, 9))
plt.imshow(R_f_split_norm, norm=colors.SymLogNorm(1e-8), cmap=cm.coolwarm)
plt.colorbar()
plt.savefig('%s/R_f_split_norm.png' % outpath)
plt.close()

plt.figure(figsize=(16, 9))
plt.imshow(np.absolute(R_f_exp), norm=colors.LogNorm())
plt.colorbar()
plt.savefig('%s/R_f_exp_amp.png' % outpath)
plt.close()

plt.figure(figsize=(16, 9))
plt.imshow(np.absolute(R_f_norm), norm=colors.LogNorm())
plt.colorbar()
plt.savefig('%s/R_f_norm_amp.png' % outpath)
plt.close()

plt.figure(figsize=(16, 9))
plt.imshow(np.angle(R_f_exp), cmap=cm.hsv)
plt.colorbar()
plt.savefig('%s/R_f_exp_phase.png' % outpath)
plt.close()

plt.figure(figsize=(16, 9))
plt.imshow(np.angle(R_f_norm), cmap=cm.hsv)
plt.colorbar()
plt.savefig('%s/R_f_norm_phase.png' % outpath)
plt.close()
