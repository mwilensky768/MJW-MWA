import numpy as np
import matplotlib.pyplot as plt
import plot_lib as pl
import rfiutil
from math import pi
from scipy.stats import chisquare, kstest

outpath = '/Users/mike_e_dubs/MWA/Test_Plots/rayl_con/'
chi_sq = np.zeros(100)
p = np.copy(chi_sq)
cutoffs = np.copy(p)
ks = np.copy(cutoffs)
p_ks = np.copy(ks)
Nbls = 8128
for N in range(1, 101):
    sim, bins, mu, var, ks[N - 1], p_ks[N - 1] = rfiutil.emp_pdf(N, Nbls)
    bin_w = np.diff(bins)
    bin_c = bins[:-1] + 0.5 * np.diff(bins)
    gauss = Nbls * bin_w[0] * np.exp(-(bin_c - mu)**2 / (2 * var)) / np.sqrt(2 * pi * var)
    chi_sq[N - 1], p[N - 1] = chisquare(sim[sim >= 1], gauss[sim >= 1], ddof=3)
    cutoffs[N - 1] = bins[-1]
    fig, ax = plt.subplots(figsize=(14, 8))
    pl.one_d_hist_plot(fig, ax, bins, [sim, gauss], labels=['Sim', 'Fit'],
                       xlog=False, xlabel='Amplitude (Median)', ylabel='Counts',
                       legend=True)
    ax.axvline(x=bins[-1], color='black')
    fig.savefig('%srayleigh_CLT_N%i.png' % (outpath, N))
    plt.close(fig)

np.save('%srayl_con_chi_sq_scipy_new_bins.npy' % (outpath), chi_sq)
np.save('%srayl_con_p_val_new_bins.npy' % (outpath), p)
np.save('%srayl_con_cutoffs.npy' % (outpath), cutoffs)
np.save('%srayl_con_ks.npy' % (outpath), ks)
np.save('%srayl_con_ks_p_values.npy' % (outpath), p_ks)

fig, ax = plt.subplots(figsize=(14, 8), nrows=2)
fig_ks, ax_ks = plt.subplots(figsize=(14, 8), nrows=2)
fig_c, ax_c = plt.subplots(figsize=(14, 8))

pl.line_plot(fig, ax[0], [chi_sq, ], title='Chi-Square per DoF',
             xlabel='N', ylabel='$\chi^2$',
             zorder=None, labels=['1e%i' % (k) for k in [6, 7]],
             legend=True, ylog=True)

pl.line_plot(fig, ax[1], [p, ], title='p-values',
             xlabel='N', ylabel='p', legend=False, ylog=False)
pl.line_plot(fig_c, ax_c, [cutoffs, ], title='KS',
             xlabel='N', ylabel='Cutoff', legend=False, ylog=False)
pl.line_plot(fig_ks, ax_ks[0], [ks, ], title='KS Test Statistic',
             xlabel='N', ylabel='KS', legend=False, ylog=True)
pl.line_plot(fig_ks, ax_ks[1], [p_ks, ], title='p-values',
             xlabel='N', ylabel='p', legend=False, ylog=False)
fig.savefig('%srayl_CLT_chi_sq_scipy_new_bins.png' % (outpath))
fig_c.savefig('%scutoffs.png' % (outpath))
fig_ks.savefig('%sKS_stat.png' % (outpath))
