import pyuvdata
import numpy as np
import matplotlib.pyplot as plt
import plot_lib
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator

FHD_dir = '/Users/mike_e_dubs/MWA/FHD/fhd_mjw_Aug23_Jan2018/'
obs_str = '1061313128_f181.2_f187.5_t30_t36'
suffixes = ['vis_XX.sav', 'vis_YY.sav', 'vis_model_XX.sav', 'vis_model_YY.sav',
            'flags.sav']
fhd_files = ['%s/vis_data/%s_%s' % (FHD_dir, obs_str, suffix) for suffix in suffixes]
fhd_files.append('%s/metadata/%s_params.sav' % (FHD_dir, obs_str))

UV1 = pyuvdata.UVData()
UV2 = pyuvdata.UVData()
UV1.read_fhd(fhd_files, use_model=True)
UV2.read_uvfits('/Users/mike_e_dubs/MWA/Data/smaller_uvfits/1061313128_f181.2_f187.5_t30_t36.uvfits')

fhd_flags = np.sum(np.reshape(UV1.flag_array, [UV2.Ntimes, UV2.Nbls, 1, UV2.Nfreqs, 2]), axis=1)
dirty_flags = np.sum(np.reshape(UV2.flag_array[:, :, :, :2], [UV2.Ntimes, UV2.Nbls, 1, UV2.Nfreqs, 2]), axis=1)

fig1, ax1 = plt.subplots(figsize=(14, 8), nrows=2)
fig1.suptitle('Model Flags')
fig2, ax2 = plt.subplots(figsize=(14, 8), nrows=2)
fig2.suptitle('Dirty Flags')

xticks = [UV2.Nfreqs * k / 5 for k in range(5)]
xticks.append(UV2.Nfreqs - 1)
xticklabels = ['%.1f' % (UV2.freq_array[0, tick] * 10 ** (-6)) for tick in xticks]
xminors = AutoMinorLocator(2)

pols = ['XX', 'YY']

for m in range(2):
    plot_lib.image_plot(fig1, ax1[m], fhd_flags[:, 0, :, m], cmap=cm.binary,
                        title=pols[m], aspect_ratio=1.6, ylabel='Time (2s)',
                        cbar_label='Baselines Flagged', xticks=xticks,
                        xticklabels=xticklabels, zero_mask=False, xminors=xminors)
    plot_lib.image_plot(fig2, ax2[m], dirty_flags[:, 0, :, m], cmap=cm.binary,
                        title=pols[m], aspect_ratio=1.6, ylabel='Time (2s)',
                        cbar_label='Baselines Flagged', xticks=xticks,
                        xticklabels=xticklabels, zero_mask=False, xminors=xminors)

fig1.savefig('/Users/mike_e_dubs/MWA/FHD/Flag_Comparison/%s_Model_Flags.png' % ('t30_t36'))
fig2.savefig('/Users/mike_e_dubs/MWA/FHD/Flag_Comparison/%s_Dirty_Flags.png' % ('t30_t36'))
