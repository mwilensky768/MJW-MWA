import numpy as np
import plot_lib
import Catalog_Funcs as cf
import matplotlib.pyplot as plt
from matplotlib import cm
import pyuvdata

# indices are time-pair/frequency/polarization
avg = np.load('/Users/mike_e_dubs/MWA/Misc/Vis_Avg_Arrays/1130773024_Vis_Avg_Amp.npy')[:, 0, :, :]

# read this in to get the frequency array
UV = pyuvdata.UVData()
UV.read_uvfits('/Users/mike_e_dubs/MWA/Data/smaller_uvfits/s1061313008.uvfits')
freqs = UV.freq_array[0, :]

# create a boolean array in order to remove the center channels from the fit by means of boolean indexing
bool_ind = np.zeros(avg.shape[1], dtype=bool)
bool_ind_centers = np.zeros(avg.shape[1], dtype=bool)
center_chans = [8 + 16 * k for k in range(24)]
LEdges = [0 + 16 * k for k in range(24)]
REdges = [15 + 16 * k for k in range(24)]
lEdges = [1 + 16 * k for k in range(24)]
lEdges2 = [2 + 16 * k for k in range(24)]
rEdges = [14 + 16 * k for k in range(24)]
rEdges2 = [13 + 16 * k for k in range(24)]

bad_chans = np.sort(np.reshape(np.array([center_chans, LEdges, REdges, lEdges,
                                         rEdges, lEdges2, rEdges2]), 24 * 7))
bad_freqs = [freqs[k] for k in bad_chans]
for k in range(len(bad_chans)):
    bool_ind[bad_chans[k]] += 1
for k in range(len(center_chans)):
    bool_ind_centers[center_chans[k]] += 1
bool_ind = np.logical_not(bool_ind)
bool_ind_centers = np.logical_not(bool_ind_centers)

# Make fit inputs and initialize fit output
x = freqs[bool_ind]
y = avg[:, bool_ind, :]
# Ci are the fit coefficients (1 and 2 are for before and after DGJ)
# polynomial fit per time pair so dims are Ntimepairs X (deg + 1) X Npols
deg = 1
C1 = np.zeros([avg.shape[0], deg + 1, avg.shape[2]])
C2 = np.zeros([avg.shape[0], deg + 1, avg.shape[2]])
fit = np.zeros(avg.shape)

# Initialize plot objects/details
fig, ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
line_fig, line_ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
ratio_fig, ratio_ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
fig.suptitle('1130773024 Deep Waterfall Excess')
line_fig.suptitle('1130773024 Deep Waterfall Excess, t = 0')
ratio_fig.suptitle('1130773024 Deep Waterfall Ratio')
xticks = [UV.Nfreqs * k / 6 for k in range(6)]
xticks.append(UV.Nfreqs - 1)
xticklabels = ['%.1f' % (10 ** (-6) * freqs[tick]) for tick in xticks]
xminors = AutoMinorLocator(4)
pol_titles = ['XX', 'YY', 'XY', 'YX']

# Make the fits
for m in range(4):
    for n in range(avg.shape[0]):
        # With bad_chans removed, there are 264 fine channels, 2/3 of which is 176
        C1[n, :, m] = np.polyfit(x[:144], y[n, :144, m], deg)
        fit[n, :256, m] = np.sum(np.array([C1[n, k, m] * freqs[:256] ** (1 - k) for k in range(1 + deg)]), axis=0)
        C2[n, :, m] = np.polyfit(x[144:], y[n, 144:, m], deg)
        fit[n, 256:, m] = np.sum(np.array([C2[n, k, m] * freqs[256:] ** (1 - k) for k in range(1 + deg)]), axis=0)

# Save the fit coefficients for later analysis...
np.save('/Users/mike_e_dubs/MWA/Misc/Vis_Avg_Arrays/1130773024_Fit_Coeff_Pre.npy', C1)
np.save('/Users/mike_e_dubs/MWA/Misc/Vis_Avg_Arrays/1130773024_Fit_Coeff_Post.npy', C2)

# Subtract the fit to leave the excess
z_excess = avg - fit
z_ratio = z_excess / avg
# Mask the coarse band center lines by setting to NaN so can ignore on the colorbar
z_excess[:, np.logical_not(bool_ind_centers), :] = float('NaN')
z_ratio[:, np.logical_not(bool_ind_centers), :] = float('NaN')
z_excess = np.ma.masked_invalid(z_excess)
z_ratio = np.ma.masked_invalid(z_ratio)


# Plot
for m in range(4):

    plot_lib.image_plot(fig, ax[m / 2][m % 2], z_excess[:, :, m],
                        title=pol_titles[m], cbar_label='Amplitude (UNCALIB)',
                        xticks=xticks, xticklabels=xticklabels,
                        mask_color='green', zero_mask=False, vmin=0)
    plot_lib.image_plot(ratio_fig, ratio_ax[m / 2][m % 2], z_ratio[:, :, m],
                        title=pol_titles[m], cbar_label='Excess/Avg',
                        xticks=xticks, xticklabels=xticklabels, mask_color='green',
                        zero_mask=False, cmap=cm.plasma, vmin=0)
    plot_lib.line_plot(line_fig, line_ax[m / 2][m % 2], [avg[0, :, m], fit[0, :, m]],
                       title=pol_titles[m], xticks=xticks, xticklabels=xticklabels,
                       zorder=[1, 2], labels=['Data', 'Fit'])


plt.show()
