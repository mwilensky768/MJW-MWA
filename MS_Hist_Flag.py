import numpy as np
import plot_lib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator
import glob
import SumThreshold as ST
from math import pi

arr_dir = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Golden_Set_8s_Autos/Vis_Avg/Averages/'
gs_list_file = '/Users/mike_e_dubs/MWA/Obs_Lists/Golden_Set_OBSIDS.txt'
freq_arr_path = '/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/MWA_Highband_Freq_Array.npy'
plot_outdir = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Templates/Golden_Set_8s_Autos/Gauss_Flag_Test_Chi/'
arr_list = glob.glob('%s*.npy' % (arr_dir))
wind_len = 32

freqs = np.load(freq_arr_path)
pols = ['XX', 'YY', 'XY', 'YX']
xticks = [len(freqs) * k / 6 for k in range(6)]
xticks.append(len(freqs) - 1)
xticklabels = ['%.1f' % (freqs[tick] * 10 ** (-6)) for tick in xticks]
xminors = AutoMinorLocator(4)

with open(gs_list_file) as g:
    obs_list = g.read().split("\n")
obs_list.remove('')

for obs in obs_list:
    INS = np.load('%s%s_Vis_Avg_Amp_All.npy' % (arr_dir, obs))[:, 0, :, :]
    mean = np.nanmean(INS, axis=0)

    frac_diff_mean = INS / mean - 1
    flag_arr = np.isnan(INS)
    not_flag_arr = np.logical_not(flag_arr)
    n, bins = np.histogram(frac_diff_mean[not_flag_arr])
    bin_wid = np.diff(bins)
    bin_cent = bins[:(len(bins) - 1)] + 0.5 * bin_wid
    N = np.sum(n)
    mu = np.mean(frac_diff_mean[not_flag_arr])
    sigma_sq = np.var(frac_diff_mean[not_flag_arr])
    fit = N * bin_wid / np.sqrt(2 * pi * sigma_sq) * np.exp(- ((bin_cent - mu) ** 2) / (2 * sigma_sq))
    chi_sq = [np.sum((n[n >= 1] - fit[n >= 1])**2 / fit[n >= 1]) / (len(fit[n >= 1]) - 3), ]

    while np.nanmax(frac_diff_mean[not_flag_arr]) > 0.022:
        flag_arr[frac_diff_mean == np.nanmax(frac_diff_mean[not_flag_arr])] = 1
        mean = np.nansum(INS * not_flag_arr, axis=0) / np.count_nonzero(not_flag_arr, axis=0)
        frac_diff_mean = INS / mean - 1
        not_flag_arr = np.logical_not(flag_arr)
        n, bins = np.histogram(frac_diff_mean[not_flag_arr])
        bin_wid = np.diff(bins)
        bin_cent = bins[:(len(bins) - 1)] + 0.5 * bin_wid
        N = np.sum(n)
        mu = np.mean(frac_diff_mean[not_flag_arr])
        sigma_sq = np.var(frac_diff_mean[not_flag_arr])
        fit = N * bin_wid / np.sqrt(2 * pi * sigma_sq) * np.exp(- ((bin_cent - mu) ** 2) / (2 * sigma_sq))
        chi_sq.append(np.sum((n[n >= 1] - fit[n >= 1])**2 / fit[n >= 1]) / (len(fit[n >= 1]) - 3))

    mean_streak = np.nanmean(frac_diff_mean, axis=1)
    for m in range(mean_streak.shape[0]):
        ind = np.where(mean_streak == np.nanmax(mean_streak))
        flag_arr[ind[0][0], :, ind[1][0]] = ST.SumThreshold(frac_diff_mean[ind[0][0], :, ind[1][0]],
                                                            flag_arr[ind[0][0], :, ind[1][0]],
                                                            wind_len, 0.022 / np.sqrt(wind_len))
        mean = np.nansum(INS * not_flag_arr, axis=0) / np.count_nonzero(not_flag_arr, axis=0)
        frac_diff_mean = INS / mean - 1
        mean_streak = np.nansum(frac_diff_mean * not_flag_arr, axis=1) / np.count_nonzero(not_flag_arr, axis=1)
        not_flag_arr = np.logical_not(flag_arr)
        n, bins = np.histogram(frac_diff_mean[not_flag_arr])
        bin_wid = np.diff(bins)
        bin_cent = bins[:(len(bins) - 1)] + 0.5 * bin_wid
        N = np.sum(n)
        mu = np.mean(frac_diff_mean[not_flag_arr])
        sigma_sq = np.var(frac_diff_mean[not_flag_arr])
        fit = N * bin_wid / np.sqrt(2 * pi * sigma_sq) * np.exp(- ((bin_cent - mu) ** 2) / (2 * sigma_sq))
        chi_sq.append(np.sum((n[n >= 1] - fit[n >= 1])**2 / fit[n >= 1]) / (len(fit[n >= 1])) - 3)

    fig, ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    fig.suptitle('%s Mean-Subtracted Incoherent Noise Spectrum, Flag Test' % (obs))
    fig_med, ax_med = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    fig_med.suptitle('%s Flagging Mask' % (obs))
    fig_chi, ax_chi = plt.subplots(figsize=(14, 8))

    for m in range(4):
        plot_lib.image_plot(fig, ax[m / 2][m % 2], frac_diff_mean[:, :, m],
                            cmap=cm.coolwarm, title=pols[m],
                            cbar_label='Fraction', xticks=xticks, xminors=xminors,
                            yminors='auto', xticklabels=xticklabels, zero_mask=False,
                            invalid_mask=True, mask_color='green')
        plot_lib.image_plot(fig_med, ax_med[m / 2][m % 2], flag_arr[:, :, m],
                            cmap=cm.binary, title=pols[m],
                            cbar_label='Fraction', xticks=xticks, xminors=xminors,
                            yminors='auto', xticklabels=xticklabels, zero_mask=False)
        plot_lib.line_plot(fig_chi, ax_chi, [chi_sq, ], title='%s Chi Square per DoF ' % (obs),
                           xlabel='Iteration', ylabel='Chi_Square per DoF',
                           zorder=[1, ], labels=['', ], legend=False)

    fig.savefig('%s%s_Mean_Subtracted_INS.png' % (plot_outdir, obs))
    fig_med.savefig('%s%s_Flags.png' % (plot_outdir, obs))
    fig_chi.savefig('%s%s_Chi.png' % (plot_outdir, obs))
    plt.close(fig)
    plt.close(fig_med)
    plt.close(fig_chi)
