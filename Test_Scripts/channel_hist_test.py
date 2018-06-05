import numpy as np
import plot_lib as pl
from matplotlib import cm, use
from matplotlib.ticker import AutoMinorLocator
import rfiutil
import matplotlib.pyplot as plt
import glob

arrpath = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Golden_Set_8s_Autos/Vis_Avg/Averages'
freqpath = '/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/MWA_Highband_Freq_Array.npy'
outpath = '/Users/mike_e_dubs/MWA/Test_Plots/channel_hist'
freqs = np.load(freqpath)
fig, ax, pols, xticks, xminors, yminors, xticklabels = pl.four_panel_tf_setup(freqs)
arrlist = glob.glob('%s/*.npy' % (arrpath))
arrlist.sort()

for arr in arrlist:
    INS = np.load(arr)
    obs = arr[len(arrpath) + 1: len(arrpath) + 11]
    hist_total, bins, hist_arr, gauss_arr, mu, var = rfiutil.channel_hist(INS)
    diff_arr = hist_arr - gauss_arr

    fig_total, ax_total = plt.subplots(figsize=(14, 8))

    pl.one_d_hist_plot(fig_total, ax_total, bins, [hist_total, ], xlog=False,
                       xlabel='Amplitude (UNCALIB)', ylabel='Counts',
                       title='%s INS Total Histogram' % (obs), legend=False)
    fig_total.savefig('%s/%s_INS_Total_Hist.png' % (outpath, obs))
    plt.close(fig_total)

    for m in range(INS.shape[1]):
        fig_hist, ax_hist = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
        fig_gauss, ax_gauss = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
        fig_diff, ax_diff = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
        for n in range(INS.shape[3]):

            pl.image_plot(fig_hist, ax_hist[n / 2][n % 2], hist_arr[:, m, :, n],
                          title=pols[n], cbar_label='Counts', xticks=xticks,
                          ylabel='Amplitude (UNCALIB)',
                          xminors=xminors, yminors=yminors, xticklabels=xticklabels)
            pl.image_plot(fig_gauss, ax_gauss[n / 2][n % 2], gauss_arr[:, m, :, n],
                          title=pols[n], cbar_label='Counts', xticks=xticks,
                          ylabel='Amplitude (UNCALIB)',
                          xminors=xminors, yminors=yminors, xticklabels=xticklabels)
            pl.image_plot(fig_diff, ax_diff[n / 2][n % 2], diff_arr[:, m, :, n],
                          cmap=cm.coolwarm, zero_mask=False, ylabel='Amplitude (UNCALIB)',
                          title=pols[n], cbar_label='Counts', xticks=xticks,
                          xminors=xminors, yminors=yminors, xticklabels=xticklabels)

        fig_hist.suptitle('%s INS Histogram' % (obs))
        fig_gauss.suptitle('%s INS Gaussian Model Histogram' % (obs))
        fig_diff.suptitle('%s INS Residual Histogram (Data - Model)' % (obs))

        fig_hist.savefig('%s/%s_INS_Hist.png' % (outpath, obs))
        fig_gauss.savefig('%s/%s_INS_Gauss_Hist.png' % (outpath, obs))
        fig_diff.savefig('%s/%s_INS_Res_Hist.png' % (outpath, obs))

        plt.close(fig_hist)
        plt.close(fig_gauss)
        plt.close(fig_diff)
