import pyuvdata
import numpy as np
import matplotlib.pyplot as plt
import plot_lib
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator

obs_str = ['t30_t36', 't18_t24']

for n in range(2):

    Cal = pyuvdata.UVCal()
    Cal.read_fhd_cal('/Users/mike_e_dubs/MWA/FHD/fhd_mjw_Aug23_Jan2018/calibration/1061313128_f181.2_f187.5_%s_cal.sav' % (obs_str[n]),
                     '/Users/mike_e_dubs/MWA/FHD/fhd_mjw_Aug23_Jan2018/metadata/1061313128_f181.2_f187.5_%s_obs.sav' % (obs_str[n]))

    UV = pyuvdata.UVData()
    UV.read_uvfits('/Users/mike_e_dubs/MWA/Data/smaller_uvfits/1061313128_f181.2_f187.5_%s.uvfits' % (obs_str[n]))

    blt_inds = [k for k in range(UV.Nblts) if UV.ant_1_array[k] != UV.ant_2_array[k]]
    UV.select(blt_inds=blt_inds)

    flags = np.reshape(UV.flag_array, [UV.Ntimes, UV.Nbls, UV.Nspws, UV.Nfreqs, UV.Npols])
    ind = np.where(flags > 0)

    ant1_ind, ant2_ind = [], []
    for inds in ind[1]:
        ant1_ind.append(UV.ant_1_array[inds])
        ant2_ind.append(UV.ant_2_array[inds])
    ant_ind = [np.array(ant1_ind), np.array(ant2_ind)]

    H = np.zeros([UV.Nants_telescope, UV.Nfreqs, UV.Npols, UV.Ntimes], dtype=int)

    for k in range(2):
        H[ant_ind[k], ind[3], ind[4], ind[0]] = 1

    H = np.mean(H, axis=-1)

    xticks = [UV.Nfreqs * k / 5 for k in range(5)]
    xticks.append(UV.Nfreqs - 1)
    xticklabels = ['%.1f' % (UV.freq_array[0, tick] * 10 ** (-6)) for tick in xticks]
    xminors = AutoMinorLocator(2)

    fig_Cal, ax_Cal = plt.subplots(nrows=2, figsize=(14, 8))
    fig_UV, ax_UV = plt.subplots(nrows=2, figsize=(14, 8))

    fig_Cal.suptitle('Calibration Flags %s' % (obs_str[n]))
    fig_UV.suptitle('COTTER Flags %s' % (obs_str[n]))

    pols = ['XX', 'YY']

    for m in range(Cal.flag_array.shape[-1]):
        plot_lib.image_plot(fig_Cal, ax_Cal[m], Cal.flag_array[:, 0, :, 0, m], cmap=cm.binary,
                            title=pols[m], aspect_ratio=0.33, ylabel='Antenna Index',
                            cbar_label='True/False', xticks=xticks,
                            xticklabels=xticklabels, zero_mask=False, xminors=xminors)

        plot_lib.image_plot(fig_UV, ax_UV[m], H[:, :, m], cmap=cm.binary,
                            title=pols[m], aspect_ratio=0.33,
                            ylabel='Antenna Index', cbar_label='True/False Average',
                            xticks=xticks, xminors=xminors, xticklabels=xticklabels,
                            zero_mask=False)

    fig_Cal.savefig('/Users/mike_e_dubs/MWA/FHD/Flag_Comparison/%s_Cal_Flags.png' % (obs_str[n]))
    fig_UV.savefig('/Users/mike_e_dubs/MWA/FHD/Flag_Comparison/%s_COTTER_Flags.png' % (obs_str[n]))
