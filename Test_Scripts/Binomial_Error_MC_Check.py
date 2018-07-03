import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import plot_lib as pl
import os

outpath = '/Users/mike_e_dubs/MWA/Test_Plots/MC_Binomial_Error'
if not os.path.exists(outpath):
    os.makedirs(outpath)

Nbins = 8
Ledge = -4
Redge = 4
bins = np.linspace(Ledge, Redge, num=Nbins + 1)
w = np.diff(bins)
x = bins[:-1] + 0.5 * w
p = norm.cdf(bins[1:]) - norm.cdf(bins[:-1])
N = 50
s = np.sqrt(N * p * (1 - p))
m = N * p
Nsim = int(1e4)
counts = np.zeros((Nsim, Nbins))

fbins = np.linspace(Ledge, Redge, num=100 * (Nbins + 1))
fw = np.diff(fbins)
fx = fbins[:-1] + 0.5 * fw
pf = norm.pdf(fx)
mf = pf * N

for i in range(Nsim):
    counts[i, :], _ = np.histogram(np.random.normal(size=50), bins=bins)

mu_sim = np.mean(counts, axis=0)
s_sim = np.sqrt(np.var(counts, axis=0))

fig, ax = plt.subplots(figsize=(14, 8))
pl.error_plot(fig, ax, x, mu_sim, None, s_sim, label='Simulation',
              drawstyle='steps-mid')
pl.error_plot(fig, ax, x, m, None, s, label='Theory', drawstyle='steps-mid')
pl.error_plot(fig, ax, fx, mf, None, None, xlabel='$\sigma$', ylabel='Counts',
              label='Standard Normal PDF')
fig.savefig('%s/Error_Plots.png' % (outpath))
plt.close(fig)

fig, ax = plt.subplots(figsize=(14, 8))
pl.line_plot(fig, ax, [s, s_sim], title='Standard Deviation Comparison',
             labels=['Theory', 'Simulations'], xlabel='$\sigma$',
             ylabel='STD (Counts)')
fig.savefig('%s/STD_Comp.png' % (outpath))
