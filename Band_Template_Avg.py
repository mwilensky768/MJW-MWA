import numpy as np
import plot_lib
import Catalog_Funcs as cf
import matplotlib.pyplot as plt
from matplotlib import cm
import pyuvdata
from matplotlib.ticker import FixedLocator, AutoMinorLocator

# indices are time-pair/frequency/polarization
avg = np.load('/Users/mike_e_dubs/MWA/Misc/Vis_Avg_Arrays/1130773024_Vis_Avg_Amp.npy')[:, 0, :, :]

# read this in to get the frequency array
UV = pyuvdata.UVData()
UV.read_uvfits('/Users/mike_e_dubs/MWA/Data/smaller_uvfits/s1061313008.uvfits')
freqs = UV.freq_array[0, :]

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
fig_exc.suptitle('1130773024 Deep Waterfall Excess')
fig_ratio.suptitle('1130773024 Deep Waterfall Ratio')
fig_line.suptitle('1130773024 Deep Waterfall Time Average')
pol_titles = ['XX', 'YY', 'XY', 'YX']
xticks = [UV.Nfreqs * k / 6 for k in range(6)]
xticks.append(UV.Nfreqs - 1)
xticklabels = ['%.1f' % (10 ** (-6) * freqs[tick]) for tick in xticks]
xminors = AutoMinorLocator(4)

# Do the plotting...
for m in range(avg.shape[2]):
    plot_lib.image_plot(fig_exc, ax_exc[m / 2][m % 2], excess[:, :, m], vmin=0,
                        title=pol_titles[m], cbar_label='Amplitude (UNCALIB)',
                        xticks=xticks, xticklabels=xticklabels, xminors=xminors,
                        zero_mask=False)
    plot_lib.image_plot(fig_ratio, ax_ratio[m / 2][m % 2], ratio[:, :, m], vmin=0,
                        title=pol_titles[m], cbar_label='Amplitude (UNCALIB)',
                        xticks=xticks, xticklabels=xticklabels, xminors=xminors,
                        zero_mask=False)
    plot_lib.line_plot(fig_line, ax_line[m / 2][m % 2], [mean[:, m], ],
                       title=pol_titles[m], xticks=xticks, xminors=xminors,
                       xticklabels=xticklabels, zorder=[1, ], labels=['Template', ])

plt.show()
