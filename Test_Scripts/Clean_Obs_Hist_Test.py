import numpy as np
import rfiutil
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
import plot_lib as pl

obs = 1061312272
indir = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Golden_Set_8s_Autos/Vis_Avg/Averages'
tag = '_Vis_Avg_Amp_All.npy'
INS = np.ma.masked_array(np.load('%s/%s%s' % (indir, obs, tag)))
C = 4 / np.pi - 1
MS = (INS / INS.mean(axis=0) - 1) * np.sqrt(8001 / C)
outdir = '/Users/mike_e_dubs/MWA/Test_Plots/Clean_Obs_Hist_Fit'
if not os.path.exists(outdir):
    os.makedirs(outdir)
pols = ['XX', 'YY', 'XY', 'YX']

for i in range(INS.shape[2]):
    fig, ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    fig.suptitle('%i MS Hist with Fit, f%i' % (obs, i))
    for k in range(INS.shape[3]):
        n, bins = np.histogram(MS[:, 0, i, k])
        fit = np.sum(n) * (norm.cdf(bins[1:], scale=1) - norm.cdf(bins[:-1], scale=1))
        pl.one_d_hist_plot(fig, ax[k / 2][k % 2], bins, [n, fit], xlog=False,
                           labels=['Measurements', 'Fit'],
                           xlabel='Mean Normalized Deviation', title=pols[k])

    fig.savefig('%s/%i_MS_hist_f%i.png' % (outdir, obs, i))
    plt.close(fig)
