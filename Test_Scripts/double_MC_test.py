import numpy as np
import plot_lib as pl
import matplotlib.pyplot as plt

shape = (51, 8001, 356)
N = 51 * 356


sim1 = np.random.rayleigh(size=shape)
scale = np.sqrt(np.random.normal(size=shape, loc=1, scale=1. / np.sqrt(51)))
# scale = np.transpose(scale, axes=(1, 2, 0))

sim2 = np.random.rayleigh(size=shape, scale=scale)

sim2_hist, bins_orig = np.histogram(sim2, bins='auto')
sim1_hist, _ = np.histogram(sim1, bins=bins_orig)

sim1_mean = sim1.mean(axis=(0, 2))
sim2_mean = sim2.mean(axis=(0, 2))

sim2_mean_hist, bins_mean = np.histogram(sim2_mean, bins='auto')
sim1_mean_hist, bins_mean_1 = np.histogram(sim1_mean, bins='auto')

fig, ax = plt.subplots(figsize=(14, 8), nrows=2)

pl.one_d_hist_plot(fig, ax[0], bins_orig, [sim1_hist, sim2_hist], labels=['s=1', 's=mixture'],
                   xlog=False, xlabel='Amplitude (Sigma)', ylabel='Counts')
pl.one_d_hist_plot(fig, ax[1], bins_mean, [sim2_mean_hist, ], labels=['s=mixture', ],
                   xlog=False, xlabel='Amplitude (Sigma)', ylabel='Counts')
pl.one_d_hist_plot(fig, ax[1], bins_mean_1, [sim1_mean_hist, ], labels=['s=1', ],
                   xlog=False, xlabel='Amplitude (Sigma)', ylabel='Counts')

fig.savefig('/Users/mike_e_dubs/MWA/Test_Plots/double_MC_rayleigh_mixture_random_quartic.png')
