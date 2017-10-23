import numpy as np
import matplotlib.pyplot as plt
import pyuvdata
from math import ceil, floor, log10
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator

cal_path = '/Users/mike_e_dubs/HERA/Temperatures/GS_Ratio_Data/Calibrated_Sigma_'
uncal_path = '/Users/mike_e_dubs/HERA/Temperatures/GS_Ratio_Data/Uncalibrated_Sigma_'
outpath = '/Users/mike_e_dubs/HERA/Temperatures/GS_Ratio_Plots/'
pols = ['XX', 'YY', 'XY', 'YX']
waterfall_title = 'HERA Golden Set Temperature Ratio (Cal / Uncal), Band Edges/Autos Removed'
cu_title = 'HERA Golden Set Temperature Ratios, for Interesting Frequencies'
N_freqs_removed = 128
UV = pyuvdata.UVData()
freq_array_obs_path = '/Users/mike_e_dubs/HERA/Data/miriad/temp_HERA_data/zen.2457555.40356.xx.HH.uvc'
UV.read_miriad(freq_array_obs_path)
N_freqs = UV.Nfreqs - N_freqs_removed
freq_chan_interval = [64, 960]
freq_array = UV.freq_array[0, min(freq_chan_interval):max(freq_chan_interval)]
temp_fig_name = 'HERA_GS_Autos_Edges_Temperatures_Ratio.png'
cu_fig_name = 'HERA_GS_Autos_Edges_Curves_Ratio.png'
curve_freqs = [160, 346, 458, 690, 881]


def sigfig(x, s=4):  # s is number of sig-figs
    if x == 0:
        return(0)
    else:
        n = int(floor(log10(np.absolute(x))))
        y = 10**n * round(10**(-n) * x, s - 1)
        return(y)


ratio = {}
for pol in pols:
    cal_sig = np.load(cal_path + pol + '.npy')
    cal_sig[cal_sig == 0] = float('nan')
    uncal_sig = np.load(uncal_path + pol + '.npy')
    uncal_sig[uncal_sig == 0] = float('nan')
    ratio[pol] = cal_sig / uncal_sig

temp_fig, temp_ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
temp_fig.suptitle(waterfall_title)
auto_pol_max = max([np.nanmax(ratio['XX']), np.nanmax(ratio['YY'])])
cross_pol_max = max([np.nanmax(ratio['XY']), np.nanmax(ratio['YX'])])
vmax = dict(zip(pols, [auto_pol_max, auto_pol_max, cross_pol_max, cross_pol_max]))
print(vmax)
xticks = [N_freqs / 6 * l for l in range(6)]
xticks.append(N_freqs - 1)
xticklabels = [str(sigfig(freq_array[k]) * 10**(-6)) for k in xticks]
for k in range(4):
    ratio[pols[k]] = np.ma.masked_equal(ratio[pols[k]], float('nan'))
    cmap = cm.plasma
    cmap.set_bad(color='white')
    cax = temp_ax[k / 2][k % 2].imshow(ratio[pols[k]], cmap=cmap, vmin=0.5, vmax=1.5)
    cbar = temp_fig.colorbar(cax, ax=temp_ax[k / 2][k % 2])
    cbar.set_label('Temperature Ratio')
    temp_ax[k / 2][k % 2].set_title(pols[k])
    temp_ax[k / 2][k % 2].set_ylabel('Observation')
    temp_ax[k / 2][k % 2].set_xlabel('Frequency (Mhz)')
    temp_ax[k / 2][k % 2].set_xticks(xticks)
    temp_ax[k / 2][k % 2].set_xticklabels(xticklabels)
    temp_ax[k / 2][k % 2].set_aspect(float(ratio[pols[k]].shape[1]) / ratio[pols[k]].shape[0])
    temp_ax[k / 2][k % 2].xaxis.set_minor_locator(AutoMinorLocator(4))


curves = {}
for pol in pols:
    for f in curve_freqs:
        curves[pol] = ratio[pol][:, f]
cu_fig, cu_ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
cu_fig.suptitle(cu_title)
for k in range(4):
    cu_ax[k / 2][k % 2].set_title(pols[k])
    cu_ax[k / 2][k % 2].set_ylabel('Temperature Ratio')
    cu_ax[k / 2][k % 2].set_xlabel('Observation')
    for f in curve_freqs:
        cu_ax[k / 2][k % 2].plot(range(len(ratio[pols[k]][:, f])),
                                 ratio[pols[k]][:, f],
                                 label=str(sigfig(freq_array[f]) * 10**(-6)) + ' Mhz')
    cu_ax[k / 2][k % 2].legend()

plt.tight_layout()

cu_fig.savefig(outpath + cu_fig_name)
temp_fig.savefig(outpath + temp_fig_name)
