import numpy as np
import plot_lib as pl
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator
import glob
import matplotlib.pyplot as plt
import os

arr_path = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Golden_Set_8s_Autos/Vis_Avg/Averages/'
outpath = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Golden_Set_8s_Autos/'
arr_list = glob.glob('%s*.npy' % (arr_path))
arr_list.sort()
freq_array = np.load('/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/MWA_Highband_Freq_Array.npy')
INS_total = np.zeros([len(arr_list), 1, len(freq_array), 4])
frac_diff_total = np.copy(INS_total)

for m, arr in enumerate(arr_list):
    INS = np.load(arr)
    INS_total[m] = INS.mean(axis=0)
    frac_diff = INS / INS_total[m] - 1
    frac_diff_total[m] = frac_diff.mean(axis=0)
    if m == 0:
        INS_stack = INS
        FD_stack = frac_diff
    else:
        INS_stack = np.vstack((INS_stack, INS))
        FD_stack = np.vstack((FD_stack, frac_diff))

INS_m = INS_total.mean(axis=0)
frac_diff_m = frac_diff_total.mean(axis=0)

fig_INS, ax_INS, pols, xticks, xminors, yminors, xticklabels = pl.four_panel_tf_setup(freq_array)
fig_fd, ax_fd = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
fig_INS_s, ax_INS_s = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
fig_FD_s, ax_FD_s = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
fig_m, ax_m = plt.subplots(figsize=(14, 8), nrows=2)
fig_mp, ax_mp = plt.subplots(figsize=(14, 8), nrows=2)

avg_title = ['Golden Set INS Averaged Across Obs', 'Golden Set Frac Diff Averaged Across Obs']
ylabels = ['Amplitude (UNCALIB)', 'Fraction']

for m in range(4):
    pl.image_plot(fig_INS, ax_INS[m / 2][m % 2], INS_total[:, 0, :, m],
                  title=pols[m], ylabel='Obs', cbar_label='Amplitude (UNCALIB)',
                  xticks=xticks, xminors=xminors, yminors=yminors,
                  xticklabels=xticklabels, zero_mask=False)
    pl.image_plot(fig_fd, ax_fd[m / 2][m % 2], frac_diff_total[:, 0, :, m],
                  cmap=cm.coolwarm, title=pols[m], xlabel='Frequency (Mhz)',
                  ylabel='Obs', cbar_label='Fraction', xticks=xticks, xminors=xminors,
                  yminors=yminors, xticklabels=xticklabels, zero_mask=False,
                  mask_color='black')
    pl.image_plot(fig_INS_s, ax_INS_s[m / 2][m % 2], INS_stack[:, 0, :, m],
                  title=pols[m], cbar_label='Amplitude (UNCALIB)', xticks=xticks,
                  aspect_ratio=0.1, xminors=xminors, xticklabels=xticklabels,
                  zero_mask=False)
    pl.image_plot(fig_FD_s, ax_FD_s[m / 2][m % 2], FD_stack[:, 0, :, m],
                  cmap=cm.coolwarm, title=pols[m], cbar_label='Fraction',
                  aspect_ratio=0.1, xticks=xticks, xminors=xminors,
                  xticklabels=xticklabels, zero_mask=False, mask_color='black')

for m in range(2):
    pl.line_plot(fig_m, ax_m[m], [[INS_m[0, :, n] for n in range(4)],
                                  [frac_diff_m[0, :, n] for n in range(4)]][m],
                 title=avg_title[m], xlabel='Frequency (Mhz)', ylabel=ylabels[m],
                 labels=pols, xticks=xticks, xticklabels=xticklabels,
                 xminors=xminors)
    pl.line_plot(fig_mp, ax_mp[m], [[INS_m[0].mean(axis=-1), ], [frac_diff_m[0].mean(axis=-1), ]][m],
                 title=avg_title[m], xlabel='Frequency (Mhz)', ylabel=ylabels[m],
                 xticks=xticks, xticklabels=xticklabels, xminors=xminors,
                 legend=False)

fig_INS.savefig('%sGolden_Set_INS.png' % (outpath))
fig_fd.savefig('%sGolden_Set_FD.png' % (outpath))
fig_m.savefig('%sGolden_Set_INS_FD_tavg.png' % (outpath))
fig_mp.savefig('%sGolden_Set_INS_FD_tpavg.png' % (outpath))
fig_INS_s.savefig('%sGolden_Set_INS_noavg.png' % (outpath))
fig_FD_s.savefig('%sGolden_Set_FD_noavg.png' % (outpath))
