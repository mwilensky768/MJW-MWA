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
with open(obslist_path) as f:
    obslist = f.read().split("\n")

# Frequency array and polarization titles and other plot objectas that are obs-independent
freqs = np.load(freq_array_path)
pol_titles = ['XX', 'YY', 'XY', 'YX']
xticks = [len(freqs) * k / 6 for k in range(6)]
xticks.append(len(freqs) - 1)
xticklabels = ['%.1f' % (10 ** (-6) * freqs[tick]) for tick in xticks]
xminors = AutoMinorLocator(4)

for obs in obslist:
    # Load deep waterfall and histogram arrays
    avg = np.load('%s%s_Vis_Avg_Amp.npy' % (avg_dir, obs))[:, 0, :, :]
    hist = np.load('%s%s_Unflagged_hist.npy' % (temp_dir, obs))

    # Average across time and construct template
    mean = np.mean(avg, axis=0)
    template = np.array([mean for m in range(avg.shape[0])])

    # Create excess and ratio data
    excess = avg - template
    ratio = excess / avg

    # Create fig/axes objects

    fig_exc, ax_exc = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    fig_ratio, ax_ratio = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    fig_line, ax_line = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    fig_exc.suptitle('%s Deep Waterfall Excess' % (obs))
    fig_ratio.suptitle('%s Deep Waterfall Ratio' % (obs))
    fig_line.suptitle('%s Deep Waterfall Time Average' % (obs))

    # Do the plotting...
    for m in range(avg.shape[2]):
        plot_lib.image_plot(fig_exc, ax_exc[m / 2][m % 2], excess[:, :, m], vmin=0,
                            title=pol_titles[m], cbar_label='Amplitude (UNCALIB)',
                            xticks=xticks, xticklabels=xticklabels, xminors=xminors,
                            zero_mask=False)
        plot_lib.image_plot(fig_ratio, ax_ratio[m / 2][m % 2], ratio[:, :, m], vmin=0,
                            title=pol_titles[m], cbar_label='Excess/Average',
                            xticks=xticks, xticklabels=xticklabels, xminors=xminors,
                            zero_mask=False)
        plot_lib.line_plot(fig_line, ax_line[m / 2][m % 2], [mean[:, m], ],
                           title=pol_titles[m], xticks=xticks, xminors=xminors,
                           xticklabels=xticklabels, zorder=[1, ], labels=['Template', ])

        fig_exc.savefig('%s%s_Vis_Avg_Excess.png' % (plot_dir, obs))
        fig_ratio.savefig('%s%s_Vis_Avg_Ratio.png' % (plot_dir, obs))
        fig_line.savefig('%s%s_Vis_Avg_Template.png' % (plot_dir, obs))
