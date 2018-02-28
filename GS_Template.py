import numpy as np
import matplotlib.pyplot as plt
import plot_lib
from matplotlib.ticker import AutoMinorLocator
from matplotlib import cm
from math import pi

rfi_list_file = '/Users/mike_e_dubs/MWA/Obs_Lists/GS_8s_Autos_INS_RFI.txt'
gs_list_file = '/Users/mike_e_dubs/MWA/Obs_Lists/Golden_Set_OBSIDS.txt'
INS_array_path = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Golden_Set_8s_Autos/Vis_Avg/Averages/'
post_flag_subdir = 'Post_Flag/Averages/'
freq_array_path = '/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/MWA_Highband_Freq_Array.npy'
plot_dir = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Templates/Golden_Set_8s_Autos/All/'
int_plot = False
hist_plot = True

#with open(rfi_list_file) as f:
    #rfi_list = f.read().split("\n")
with open(gs_list_file) as g:
    obs_list = g.read().split("\n")

obs_list.remove('')

#for obs in rfi_list:
    #obs_list.remove(obs)

fig, ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
fig_med, ax_med = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
pols = ['XX', 'YY', 'XY', 'YX']

freqs = np.load(freq_array_path)
xticks = [len(freqs) * k / 6 for k in range(6)]
xticks.append(len(freqs) - 1)
xticklabels = ['%.1f' % (freqs[tick] * 10 ** (-6)) for tick in xticks]
xminors = AutoMinorLocator(4)

for obs in obs_list:
    INS = np.load('%s%s_Vis_Avg_Amp_All.npy' % (INS_array_path, obs))[:, 0, :, :]
    mean = np.mean(INS, axis=0)
    median = np.median(INS, axis=0)
    template = np.array([mean for k in range(INS.shape[0])])
    med_template = np.array([median for k in range(INS.shape[0])])

    INS_PF = np.load('%s%s%s_Vis_Avg_Amp_Unflagged.npy' % (INS_array_path, post_flag_subdir, obs))[:, 0, :, :]
    mean_pf = np.nanmean(INS_PF, axis=0)
    median_pf = np.nanmedian(INS_PF, axis=0)
    template_pf = np.array([mean_pf for k in range(INS.shape[0])])
    med_template_pf = np.array([median_pf for k in range(INS.shape[0])])

    Frac_Diff = (INS - template) / template
    Frac_Diff_med = (INS - med_template) / med_template
    Frac_Diff_pf = (INS_PF - template_pf) / template_pf
    Frac_Diff_pf_med = (INS_PF - med_template_pf) / med_template_pf

    fig, ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    fig_med, ax_med = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    fig_pf, ax_pf = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    fig_pf_med, ax_pf_med = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    fig_hist, ax_hist = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)

    fig.suptitle('%s Incoherent Noise Spectrum Excess from Mean' % (obs))
    fig_med.suptitle('%s Incoherent Noise Spectrum Excess from Median' % (obs))
    fig_pf.suptitle('%s Incoherent Noise Spectrum Excess from Mean, Post-Flagging' % (obs))
    fig_pf_med.suptitle('%s Incoherent Noise Spectrum Excess from Median, Post-Flagging' % (obs))
    fig_hist.suptitle('%s Incoherent Noise Spectrum Fractional Deviation Histogram' % (obs))

    for m in range(4):

        n, bins = np.histogram(Frac_Diff[:, :, m], bins='auto')
        widths = np.diff(bins)
        centers = bins[:-1] + 0.5 * widths
        wind_cond = np.abs(Frac_Diff[:, :, m] < 0.02)
        var = np.mean((Frac_Diff[:, :, m]**2)[wind_cond])
        avg = np.mean(Frac_Diff[:, :, m][wind_cond])
        N = len(Frac_Diff[:, :, m][wind_cond])
        dist = N * widths / np.sqrt(2 * pi * var) * np.exp(-(centers - avg)**2 / (2 * var))

        if int_plot:
            plot_lib.image_plot(fig, ax[m / 2][m % 2], Frac_Diff[:, :, m],
                                cmap=cm.coolwarm, title=pols[m], cbar_label='Fraction',
                                xticks=xticks, xticklabels=xticklabels, xminors=xminors,
                                zero_mask=False, yminors='auto')

            plot_lib.image_plot(fig_med, ax_med[m / 2][m % 2], Frac_Diff_med[:, :, m],
                                cmap=cm.coolwarm, title=pols[m], cbar_label='Fraction',
                                xticks=xticks, xticklabels=xticklabels, xminors=xminors,
                                zero_mask=False, yminors='auto')

            plot_lib.image_plot(fig_pf, ax_pf[m / 2][m % 2], Frac_Diff_pf[:, :, m],
                                cmap=cm.coolwarm, title=pols[m], cbar_label='Fraction',
                                xticks=xticks, xticklabels=xticklabels, xminors=xminors,
                                zero_mask=False, yminors='auto', invalid_mask=True)

            plot_lib.image_plot(fig_pf_med, ax_pf_med[m / 2][m % 2], Frac_Diff_pf_med[:, :, m],
                                cmap=cm.coolwarm, title=pols[m], cbar_label='Fraction',
                                xticks=xticks, xticklabels=xticklabels, xminors=xminors,
                                zero_mask=False, yminors='auto', invalid_mask=True)

        if hist_plot:
            plot_lib.one_d_hist_plot(fig_hist, ax_hist[m / 2][m % 2], bins, [n, dist],
                                     zorder=[1, 2], labels=['Histogram', 'Gaussian Fit'],
                                     legend=True, title=pols[m],
                                     xlog=False, xlabel='Fractional Deviation')

    if int_plot:
        fig.savefig('%s%s_INS_Mean_Excess.png' % (plot_dir, obs))
        fig_med.savefig('%s%s_INS_Median_Excess.png' % (plot_dir, obs))
        fig_pf_med.savefig('%s%s_INS_Median_Excess_Post_Flag.png' % (plot_dir, obs))
        fig_pf.savefig('%s%s_INS_Mean_Excess_Post_Flag.png' % (plot_dir, obs))
    if hist_plot:
        fig_hist.savefig('%s%s_INS_Mean_Excess_Hist.png' % (plot_dir, obs))

    plt.close(fig)
    plt.close(fig_med)
    plt.close(fig_pf)
    plt.close(fig_pf_med)
    plt.close(fig_hist)
