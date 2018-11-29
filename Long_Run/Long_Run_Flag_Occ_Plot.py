import numpy as np
import pickle
from SSINS import plot_lib
import matplotlib.pyplot as plt
from matplotlib import cm
import os

basedir = '/Users/mike_e_dubs/MWA/INS/Long_Run/Original_Jackknife'
metadir = '/Users/mike_e_dubs/MWA/INS/Long_Run/time_arrs'
occ_dict = pickle.load(open('%s/long_run_original_occ_dict.pik' % basedir, 'rb'))
sig_list = [5, 10, 20, 40, 80]
edges = [16 * k for k in range(24)] + [15 + 16 * k for k in range(24)]
good_freqs = np.ones(384, dtype=bool)
good_freqs[edges] = 0


for sig_thresh in sig_list:

    x_data = []
    y_data = []
    c = []
    c_TV7 = []
    c_broad7 = []

    for obs in occ_dict[sig_thresh]['point']:
        if os.path.exists('%s/%s_lst_arr.npy' % (metadir, obs)):
            x = np.load('%s/%s_lst_arr.npy' % (metadir, obs))[0]
            if x > np.pi:
                x -= 2 * np.pi
            x *= 23.9345 / (2 * np.pi)
            x_data.append(x)

            y = np.load('%s/%s_times_arr.npy' % (metadir, obs))[0]
            y_data.append(y)
            occ_dat = occ_dict[sig_thresh]['point'][obs][good_freqs]
            occ_TV7 = occ_dict[sig_thresh]['TV7'][obs]
            occ_broad7 = occ_dict[sig_thresh]['broad7'][obs]
            c.append(np.mean(occ_dat))
            c_TV7.append(occ_TV7)
            c_broad7.append(occ_broad7)
    fig_hist, ax_hist = plt.subplots(figsize=(8, 4.5))
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    c = np.array(c)
    c_TV7 = np.array(c_TV7)
    c_broad7 = np.array(c_broad7)
    where7 = c_TV7 == 0
    where_broad = c_broad7 == 0
    c_tv = np.copy(c_TV7)
    c_tv[where7] = c_broad7[where7]
    counts, bins = np.histogram(c, bins=np.linspace(0, 1, num=50))
    Nz = len(c[c == 0])
    counts = np.append(counts, 0)
    plot_lib.error_plot(fig_hist, ax_hist, bins, counts, drawstyle='steps-post',
                        title='Total Occupation Histogram %i$\hat{\sigma}$' % sig_thresh,
                        xlabel='Occupation Fraction', ylabel='Counts', ylim=[0.1, 10**3],
                        yscale='log', label='$N_z = $ %i' % Nz, legend=True, leg_size='xx-large')
    fig_hist.savefig('%s/%is_hist_total_occ.png' % (basedir, sig_thresh))
    plt.close(fig_hist)

    fig_scat, ax_scat = plt.subplots(figsize=(8, 4.5))

    plot_lib.scatter_plot_2d(fig_scat, ax_scat, x_data[c == 0], y_data[c == 0], vmax=0.4,
                             title='Total Occupation Scatter %i$\hat{\sigma}$' % sig_thresh,
                             xlabel='LST (hours)', ylabel='Julian Date (days)',
                             c='white', edgecolors='brown', cbar_label='Occupation Fraction',
                             s=10)
    plot_lib.scatter_plot_2d(fig_scat, ax_scat, x_data[c > 0], y_data[c > 0], vmax=0.4,
                             title='Total Occupation Scatter %i$\hat{\sigma}$' % sig_thresh,
                             xlabel='LST (hours)', ylabel='Julian Date (days)',
                             c=c[c > 0], cmap=cm.copper_r, cbar_label='Occupation Fraction',
                             s=10)
    fig_scat.savefig('%s/%is_scat_total_occ.png' % (basedir, sig_thresh))
    plt.close(fig_scat)

    fig_TV7, ax_TV7 = plt.subplots(figsize=(8, 4.5))

    plot_lib.scatter_plot_2d(fig_TV7, ax_TV7, x_data[c_TV7 == 0], y_data[c_TV7 == 0], vmax=0.4,
                             title='TV7 Occupation Scatter %i$\hat{\sigma}$' % sig_thresh,
                             xlabel='LST (hours)', ylabel='Julian Date (days)',
                             c='white', edgecolors='brown', cbar_label='Occupation Fraction',
                             s=10)
    plot_lib.scatter_plot_2d(fig_TV7, ax_TV7, x_data[c_TV7 > 0], y_data[c_TV7 > 0], vmax=0.4,
                             title='TV7 Occupation Scatter %i$\hat{\sigma}$' % sig_thresh,
                             xlabel='LST (hours)', ylabel='Julian Date (days)',
                             c=c_TV7[c_TV7 > 0], cmap=cm.copper_r, cbar_label='Occupation Fraction',
                             s=10)
    fig_TV7.savefig('%s/%is_scat_TV7_occ.png' % (basedir, sig_thresh))
    plt.close(fig_TV7)

    fig_broad7, ax_broad7 = plt.subplots(figsize=(8, 4.5))

    plot_lib.scatter_plot_2d(fig_broad7, ax_broad7, x_data[c_broad7 == 0], y_data[c_broad7 == 0], vmax=0.4,
                             title='Broad7 Occupation Scatter %i$\hat{\sigma}$' % sig_thresh,
                             xlabel='LST (hours)', ylabel='Julian Date (days)',
                             c='white', edgecolors='brown', cbar_label='Occupation Fraction',
                             s=10)
    plot_lib.scatter_plot_2d(fig_broad7, ax_broad7, x_data[c_broad7 > 0], y_data[c_broad7 > 0], vmax=0.4,
                             title='Broad7 Occupation Scatter %i$\hat{\sigma}$' % sig_thresh,
                             xlabel='LST (hours)', ylabel='Julian Date (days)',
                             c=c_broad7[c_broad7 > 0], cmap=cm.copper_r, cbar_label='Occupation Fraction',
                             s=10)
    fig_broad7.savefig('%s/%is_scat_broad7_occ.png' % (basedir, sig_thresh))
    plt.close(fig_broad7)

    fig_tv, ax_tv = plt.subplots(figsize=(8, 4.5))

    plot_lib.scatter_plot_2d(fig_tv, ax_tv, x_data[c_tv == 0], y_data[c_tv == 0], vmax=0.4,
                             title='TV7 Occupation Scatter %i$\hat{\sigma}$' % sig_thresh,
                             xlabel='LST (hours)', ylabel='Julian Date (days)',
                             c='white', edgecolors='brown', cbar_label='Occupation Fraction',
                             s=10)
    plot_lib.scatter_plot_2d(fig_tv, ax_tv, x_data[c_tv > 0], y_data[c_tv > 0], vmax=0.4,
                             title='TV7 Occupation Scatter %i$\hat{\sigma}$' % sig_thresh,
                             xlabel='LST (hours)', ylabel='Julian Date (days)',
                             c=c_tv[c_tv > 0], cmap=cm.copper_r, cbar_label='Occupation Fraction',
                             s=10)
    fig_tv.savefig('%s/%is_scat_tv_occ.png' % (basedir, sig_thresh))
    plt.close(fig_tv)
