wimport rfiutil
import numpy as np
import scipy.stats
import plot_lib as pl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator
import os

obs = 1061312272
arr_path = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Golden_Set_8s_Autos/Vis_Avg/Averages/%i_Vis_Avg_Amp_All.npy' % (obs)
pols = ['XX', 'YY', 'XY', 'YX']
freq_arr_path = '/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/MWA_Highband_Freq_array.npy'
freqs = np.load(freq_arr_path)
mode = 'approx'
xticks = [64 * i for i in range(6)] + [383, ]
xticklabels = ['%.1f' % (freqs[tick] * 10 ** (-6)) for tick in xticks]
outpath = '/Users/mike_e_dubs/MWA/Test_Plots/KS_Test/%s' % (mode)
if not os.path.exists(outpath):
    os.makedirs(outpath)
INS = np.ma.masked_array(np.load(arr_path))
C = 4 / np.pi - 1
logbins = True


for sig_thresh in range(1, 11):
    MS = np.sqrt(8001 / C) * (INS / INS.mean(axis=0) - 1)
    sim = np.ma.masked_array(np.random.normal(size=INS.shape))
    ks_arr = rfiutil.ks_test(MS, mode=mode)
    ks_arr_sim = rfiutil.ks_test(sim, mode=mode)
    D = []
    p = []
    D_sim = []
    p_sim = []
    alpha = 2 * (1 - scipy.stats.norm.cdf(sig_thresh))
    for i in range(INS.shape[1]):
        for k in range(INS.shape[2]):
            for m in range(INS.shape[3]):
                D.append(ks_arr[i, k, m][0])
                p.append(ks_arr[i, k, m][1])
                D_sim.append(ks_arr_sim[i, k, m][0])
                p_sim.append(ks_arr_sim[i, k, m][1])
                if ks_arr[i, k, m][1] < alpha:
                    MS[:, i, k, m] = np.ma.masked
    if logbins:
        bins = np.logspace(-3, 0, num=100)
        D_hist, D_bins = np.histogram(D, bins=bins)
        p_hist, p_bins = np.histogram(p, bins=bins)
    else:
        D_hist, D_bins = np.histogram(D, bins='auto')
        p_hist, p_bins = np.histogram(p, bins='auto')
    D_hist_sim, _ = np.histogram(D_sim, bins=D_bins)
    p_hist_sim, _ = np.histogram(p_sim, bins=p_bins)
    hists = [[D_hist, D_hist_sim], [p_hist, p_hist_sim]]
    bins = [D_bins, p_bins]
    fig, ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    fig.suptitle('%i ks test, $\sigma$=%i' % (obs, sig_thresh))
    kwargs = {'cmap': cm.coolwarm,
              'xlabel': 'Frequency (Mhz)',
              'cbar_label': '$\sigma$',
              'xticks': xticks,
              'xticklabels': xticklabels,
              'xminors': AutoMinorLocator(4),
              'mask_color': 'black'}
    for i in range(4):
        args = (fig, ax[i / 2][i % 2], MS[:, 0, :, i])
        kwargs['title'] = pols[i]
        pl.image_plot(*args, **kwargs)
        fig.savefig('%s/%i_KS_%i.png' % (outpath, obs, sig_thresh))
    plt.close(fig)
    titles = ['KS Statistic Histogram', 'KS P-Value Histogram']
    tags = ['KS_Stat', 'KS_P']
    xlabels = ['$D_n$', 'p']
    for hist, bin, title, tag, xlabel in zip(hists, bins, titles, tags, xlabels):
        fig, ax = plt.subplots(figsize=(14, 8))
        args = (fig, ax, bin, hist)
        kwargs = {'xlog': True,
                  'ylog': True,
                  'xlabel': xlabel,
                  'title': title,
                  'legend': True}
        pl.one_d_hist_plot(*args, labels=['Measurements', 'Simulation'], **kwargs)
        fig.savefig('%s/%i_%s_loglog.png' % (outpath, obs, tag))
        plt.close(fig)
