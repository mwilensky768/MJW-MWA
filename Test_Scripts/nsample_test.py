from pyuvdata import UVData
import numpy as np
import plot_lib
from matplotlib import cm

UV = UVData()
UV.read_uvfits('/Users/mike_e_dubs/MWA/Data/uvfits/1061313128.uvfits')

ind_0 = np.where(UV.nsample_array == 0)
cond = np.logical_and(UV.nsample_array > 0, UV.flag_array).reshape([UV.Ntimes,
                                                                    UV.Nbls,
                                                                    UV.Nspws,
                                                                    UV.Nfreqs,
                                                                    UV.Npols]).sum(axis=1)

fig, ax, pols, xticks, xminors, yminors, xticklabels = plot_lib.four_panel_tf_setup(UV.freq_array[0, :])

for m in range(4):
    plot_lib.image_plot(fig, ax[m / 2][m % 2], cond[:, 0, :, m], cmap=cm.plasma,
                        title=pols[m], ylabel='Time (2s)',
                        cbar_label='Nbls', xticks=xticks, xminors=xminors,
                        yminors=yminors, xticklabels=xticklabels,
                        mask_color='white')

fig.savefig('/Users/mike_e_dubs/MWA/Test_Plots/nsample_test/1061313128_nsample_where_flag.png')
