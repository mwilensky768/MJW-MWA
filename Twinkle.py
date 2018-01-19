import pyuvdata
import numpy as np
import plot_lib
import matplotlib.pyplot as plt
from matplotlib import cm
from math import ceil, floor, log10
from matplotlib.ticker import FixedLocator, AutoMinorLocator

# Read in the data
obspath = '/Users/mike_e_dubs/MWA/Data/smaller_uvfits/1061313128f_180_190.uvfits'
figpath = '/Users/mike_e_dubs/MWA/Catalogs/Twinkle/Diff(Avg(Abs))/'
UV = pyuvdata.UVData()
UV.read_uvfits(obspath)

# Remove the autocorrelations
blt_inds = [k for k in range(UV.Nblts) if UV.ant_1_array[k] != UV.ant_2_array[k]]
UV.select(blt_inds=blt_inds)

# Construct the template
mean = np.mean(np.absolute(np.reshape(UV.data_array, [UV.Ntimes, UV.Nbls, UV.Nspws,
                                      UV.Nfreqs, UV.Npols])), axis=0)
template = np.array([mean for k in range(UV.Ntimes)])

# Take the differences
diff = np.absolute(np.reshape(UV.data_array, [UV.Ntimes, UV.Nbls, UV.Nspws,
                                              UV.Nfreqs, UV.Npols])) - template

# Make the histograms
bins = np.logspace(floor(log10(np.amin(diff[diff > 0]))),
                   ceil(log10(np.amax(diff))), num=1001)
N = np.prod(diff.shape)
values = np.reshape(diff, N)
# flags = np.reshape(UV.flag_array, N)
# values = values[flags > 0]
m, bins = np.histogram(values, bins=bins)
ind = np.digitize(diff, bins)

# Make the baseline Average
diff_avg = np.mean(diff, axis=1)

# Make the plots
fig, ax = plt.subplots(figsize=(14, 8))
plot_lib.one_d_hist_plot(fig, ax, bins, [m, ], labels=['counts', ], zorder=[1, ],
                         title='Visibility Difference Amplitude Histogram, Diff(Avg(Abs))')
fig.savefig('%sTwinkle_One_D_Hist.png' % (figpath))


fig_dw, ax_dw = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
fig_dw.suptitle('1061313128 Time-Average Difference Deep Waterfall, Diff(Avg(Abs))')
pols = ['XX', 'YY', 'XY', 'YX']
xticks = [UV.Nfreqs * k / 8 for k in range(8)]
xticks.append(UV.Nfreqs - 1)
xticklabels = ['%.1f' % (UV.freq_array[0, k] * 10 ** (-6)) for k in xticks]
xminors = AutoMinorLocator(4)
for n in range(1000):
    fig_wh, ax_wh = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    fig_wh.suptitle('1061313128 Time-Average Difference Waterfall Histogram, bin %i, Diff(Avg(Abs))' % (n))
    H = np.zeros(diff.shape)
    H[np.where(ind == n + 1)] = 1
    H = np.sum(H, axis=1)
    for m in range(4):
        plot_lib.image_plot(fig_wh, ax_wh[m / 2][m % 2], H[:, 0, :, m], cmap=cm.cool,
                            title=pols[m], ylabel='Time (2s)', xticks=xticks,
                            xticklabels=xticklabels, xminors=xminors,
                            aspect_ratio=1)

    fig_wh.savefig('%sTwinkle_Waterfall_Hist_%i.png' % (figpath, n))
    plt.close(fig_wh)

for m in range(4):
    plot_lib.image_plot(fig_dw, ax_dw[m / 2][m % 2], diff_avg[:, 0, :, m],
                        title=pols[m], ylabel='Time (2s)',
                        cbar_label='Amplitude (%s)' % (UV.vis_units),
                        xticks=xticks, xticklabels=xticklabels, xminors=xminors,
                        aspect_ratio=1)

fig_dw.savefig('%sTwinkle_Deep_Waterfall.png' % (figpath))
