import numpy as np
import plot_lib
import Catalog_Funcs as cf
import matplotlib.pyplot as plt
from matplotlib import cm
import pyuvdata
from matplotlib.ticker import FixedLocator, AutoMinorLocator

# Input information
obslist_path = '/Users/mike_e_dubs/MWA/Obs_Lists/Diffuse_2015_GP_10s_Autos_RFI_Free.txt'
freq_array_path = '/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/MWA_Highband_Freq_Array.npy'
avg_dir = '/Users/mike_e_dubs/MWA/Temperatures/Diffuse_2015_10s_Autos/RFI_Free/averages/'
temp_dir = '/Users/mike_e_dubs/MWA/Temperatures/Diffuse_2015_10s_Autos/RFI_Free/temperatures/'
plot_dir = '/Users/mike_e_dubs/MWA/Catalogs/Diffuse_2015_10s_Autos/Vis_Avg/Template/RFI_Free/'
out_array_dir = '/Users/mike_e_dubs/MWA/Temperatures/Diffuse_2015_10s_Autos/RFI_Free/max_fit_arrays/'
with open(obslist_path) as f:
    obslist = f.read().split("\n")

obslist.remove('')

# Frequency array and polarization titles and other plot objectas that are obs-independent
freqs = np.load(freq_array_path)
pol_titles = ['XX', 'YY', 'XY', 'YX']
xticks = [len(freqs) * k / 6 for k in range(6)]
xticks.append(len(freqs) - 1)
xticklabels = ['%.1f' % (10 ** (-6) * freqs[tick]) for tick in xticks]
xminors = AutoMinorLocator(4)

# These data will be used to make a scaling parameter scheme...
max_loc_array = np.zeros(len(obslist))
fit_coeff_array = np.zeros([len(obslist), 2, 2, 4])
fit_centers_coeff_array = np.zeros([len(obslist), 2, 2, 4])
fit_edges_coeff_array = np.zeros([len(obslist), 2, 2, 4])

for n in range(len(obslist)):
    # Load deep waterfall and histogram arrays
    avg = np.load('%s%s_Vis_Avg_Amp.npy' % (avg_dir, obslist[n]))[:, 0, :, :]
    temps = np.transpose(np.array([np.load('%s%s_All_sigma_%s.npy' %
                                  (temp_dir, obslist[n], pol_titles[m])) for m in range(avg.shape[2])]))
    hist = np.load('%s%s_All_hist.npy' % (temp_dir, obslist[n]))
    bins = np.load('%s%s_All_bins.npy' % (temp_dir, obslist[n]))

    # Average across time and construct template
    mean = np.mean(avg, axis=0)
    template = np.array([mean for m in range(avg.shape[0])])

    # Find where the vis difference histogram has a maximum
    max_loc_array[n] = min(bins[:-1][hist == np.amax(hist)])

    # Will eventually construct a linear fit for the template, ignoring coarse band centers and edges
    # create a boolean array in order to remove the center channels from the fit by means of boolean indexing
    bool_ind = np.zeros(avg.shape[1], dtype=bool)
    bool_ind_centers = np.zeros(avg.shape[1], dtype=bool)
    bool_ind_edges = np.zeros(avg.shape[1], dtype=bool)
    center_chans = [8 + 16 * k for k in range(24)]
    LEdges = [0 + 16 * k for k in range(24)]
    REdges = [15 + 16 * k for k in range(24)]
    lEdges = [1 + 16 * k for k in range(24)]
    lEdges2 = [2 + 16 * k for k in range(24)]
    rEdges = [14 + 16 * k for k in range(24)]
    rEdges2 = [13 + 16 * k for k in range(24)]

    bad_chans = np.sort(np.reshape(np.array([center_chans, LEdges, REdges, lEdges,
                                             rEdges, lEdges2, rEdges2]), 24 * 7))
    edge_chans = np.sort(np.reshape(np.array([LEdges, REdges]), 24 * 2))

    bad_freqs = [freqs[k] for k in bad_chans]
    for k in bad_chans:
        bool_ind[k] += 1
    for k in center_chans:
        bool_ind_centers[k] += 1
    for k in edge_chans:
        bool_ind_edges[k] += 1
    bool_ind = np.logical_not(bool_ind)

    # Make fit inputs and initialize fit output
    x = freqs[bool_ind]
    y = mean[bool_ind, :]
    x_centers = freqs[bool_ind_centers]
    x_edges = freqs[bool_ind_edges]

    # polynomial fit per time pair so dims are Ntimepairs X (deg + 1) X Npols
    deg = 1
    fit = np.zeros(mean.shape)
    fit_centers = np.zeros(mean.shape)
    fit_edges = np.zeros(mean.shape)

    # Make the fits
    for m in range(avg.shape[2]):
        # With bad_chans removed, there are 264 fine channels, 2/3 of which is 176
        # For this data set do not need the DGJ...
        fit_coeff_array[n, 0, :, m] = np.polyfit(x, y[:, m], deg)
        fit[:, m] = np.sum(np.array([fit_coeff_array[n, 0, k, m] * freqs ** (1 - k) for k in range(1 + deg)]), axis=0)
        y_centers = mean[bool_ind_centers, m] - fit[bool_ind_centers, m]
        y_edges = mean[bool_ind_edges, m] - fit[bool_ind_edges, m]
        fit_centers_coeff_array[n, 0, :, m] = np.polyfit(x_centers, y_centers, deg)
        fit_edges_coeff_array[n, 0, :, m] = np.polyfit(x_edges, y_edges, deg)
        fit_centers[:, m] = np.sum(np.array([fit_centers_coeff_array[n, 0, k, m] * freqs ** (1 - k) for k in range(1 + deg)]), axis=0)
        fit_edges[:, m] = np.sum(np.array([fit_edges_coeff_array[n, 0, k, m] * freqs ** (1 - k) for k in range(1 + deg)]), axis=0)

    # Create excess, ratio, and residual data
    excess = avg - template
    ratio = excess / avg

    # Create fig/axes objects

    fig_exc, ax_exc = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    fig_ratio, ax_ratio = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    fig_line, ax_line = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    fig_scatter, ax_scatter = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    fig_exc.suptitle('%s Deep Waterfall Excess' % (obslist[n]))
    fig_ratio.suptitle('%s Deep Waterfall Ratio' % (obslist[n]))
    fig_line.suptitle('%s Deep Waterfall Time Average' % (obslist[n]))
    fig_scatter.suptitle('%s Mean vs. Rayleigh Fit Correlation' % (obslist[n]))

    # Do the plotting...
    for m in range(avg.shape[2]):
        plot_lib.image_plot(fig_exc, ax_exc[m / 2][m % 2], excess[:, :, m],
                            title=pol_titles[m], cbar_label='Amplitude (UNCALIB)',
                            xticks=xticks, xticklabels=xticklabels, xminors=xminors,
                            zero_mask=False, cmap=cm.coolwarm)
        plot_lib.image_plot(fig_ratio, ax_ratio[m / 2][m % 2], ratio[:, :, m],
                            title=pol_titles[m], cbar_label='Excess/Average',
                            xticks=xticks, xticklabels=xticklabels, xminors=xminors,
                            zero_mask=False, cmap=cm.coolwarm)
        plot_lib.line_plot(fig_line, ax_line[m / 2][m % 2],
                           [mean[:, m], fit[:, m], fit[:, m] + fit_centers[:, m],
                           fit[:, m] + fit_edges[:, m]],
                           title=pol_titles[m], xticks=xticks, xminors=xminors,
                           xticklabels=xticklabels, zorder=[1, 2, 2, 2],
                           labels=['Template', 'Fit', 'Center Teeth Fit',
                                   'Edge Teeth Fit'])
        plot_lib.scatter_plot_2d(fig_scatter, ax_scatter[m / 2][m % 2], temps[:, m], mean[:, m],
                                 title=pol_titles[m], xlabel='Fit Width',
                                 ylabel='Template')

    fig_exc.savefig('%s%s_Vis_Avg_Excess.png' % (plot_dir, obslist[n]))
    fig_ratio.savefig('%s%s_Vis_Avg_Ratio.png' % (plot_dir, obslist[n]))
    fig_line.savefig('%s%s_Vis_Avg_Template.png' % (plot_dir, obslist[n]))
    fig_scatter.savefig('%s%s_Vis_Avg_Temperature.png' % (plot_dir, obslist[n]))

    plt.close(fig_exc)
    plt.close(fig_ratio)
    plt.close(fig_line)
    plt.close(fig_scatter)

np.save('%smax_loc_array.npy' % (out_array_dir), max_loc_array)
np.save('%sfit_coeff_array.npy' % (out_array_dir), fit_coeff_array)
np.save('%sfit_centers_coeff_array' % (out_array_dir), fit_centers_coeff_array)
np.save('%sfit_edges_coeff_array' % (out_array_dir), fit_edges_coeff_array)
