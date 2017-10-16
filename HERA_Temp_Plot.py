import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pyuvdata
from math import floor, ceil, log10
from matplotlib.ticker import AutoMinorLocator
import glob

obs_pathlist = glob.glob('/Users/mike_e_dubs/python_stuff/miriad/temp_HERA_data/*.uvc')
inpath = '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Temperatures_HERA/Hists_Midband_Autos/'
outpath = '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Temperatures_HERA/Plots_Midband_Autos/'
N_freqs_removed = 1024 - 100
UV = pyuvdata.UVData()
UV.read_miriad(obs_pathlist[0])
N_freqs = UV.Nfreqs - N_freqs_removed
freq_chan_interval = [550, 650]
freq_array = UV.freq_array[0, min(freq_chan_interval):max(freq_chan_interval)]


def sigfig(x, s=4):  # s is number of sig-figs
    if x == 0:
        return(0)
    else:
        n = int(floor(log10(np.absolute(x))))
        y = 10**n * round(10**(-n) * x, s - 1)
        return(y)


pols = ['XX', 'YY', 'XY', 'YX']

sigma = {}
for pol in pols:
    sigma[pol] = np.zeros([len(obs_pathlist) / 4, N_freqs])

k = 0
for path in obs_pathlist:
    start = path.find('zen.')
    end = path.find('.uvc')
    obs = path[start:end]
    pol = path[end - 5:end - 3].upper()
    if k == 0:
        n = np.load(inpath + obs + '_hist.npy')
        bins = np.load(inpath + obs + '_bins.npy')
        fit = np.load(inpath + obs + '_fit.npy')
        sigma[pol][k / 4, :] += np.load(inpath + obs + '_sigma_' + pol + '.npy')
    else:
        n += np.load(inpath + obs + '_hist.npy')
        fit += np.load(inpath + obs + '_fit.npy')
        sigma[pol][k / 4, :] += np.load(inpath + obs + '_sigma_' + pol + '.npy')
    k += 1

residual = n - fit

widths = np.diff(bins)
centers = bins[:-1] + 0.5 * widths
N = len(n)

hist_fig, hist_ax = plt.subplots(figsize=(14, 8), nrows=2)
hist_ax[0].step(bins[:-1], n, where='pre', label='Histogram (No Flag Cut)')
hist_ax[0].plot(centers, fit, label='Fit')
hist_ax[0].set_title('Visibility Difference Histogram, HERA Preliminiary, Band Edges/Autos Removed')
hist_ax[0].set_xlabel('Amplitude (UNCALIB)')
hist_ax[0].set_ylabel('Counts')
hist_ax[0].set_xscale('log', nonposy='clip')
hist_ax[0].set_yscale('log', nonposy='clip')
hist_ax[0].set_ylim([10 ** (-1), 10 ** (12)])
hist_ax[0].legend()
hist_ax[1].plot(centers, residual, label='Residual')
hist_ax[1].set_xscale('log', nonposy='clip')
hist_ax[1].set_yscale('linear')
hist_ax[1].set_xlabel('Amplitude (UNCALIB)')
hist_ax[1].set_ylabel('Counts')
hist_ax[1].legend()

temp_fig, temp_ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
temp_fig.suptitle('HERA Preliminary Temperatures, ~ 155-165 Mhz /Autos Removed')
auto_pol_max = max([np.amax(sigma['XX']), np.amax(sigma['YY'])])
cross_pol_max = max([np.amax(sigma['XY']), np.amax(sigma['YX'])])
vmax = dict(zip(pols, [auto_pol_max, auto_pol_max, cross_pol_max, cross_pol_max]))
xticks = [N_freqs / 6 * l for l in range(6)]
xticks.append(N_freqs - 1)
xticklabels = [str(sigfig(freq_array[k]) * 10**(-6)) for k in xticks]
for k in range(4):
    sigma[pols[k]] = np.ma.masked_equal(sigma[pols[k]], 0)
    cmap = cm.cool
    cmap.set_bad(color='white')
    cax = temp_ax[k / 2][k % 2].imshow(sigma[pols[k]], cmap=cmap, vmin=0, vmax=vmax[pols[k]])
    cbar = temp_fig.colorbar(cax, ax=temp_ax[k / 2][k % 2])
    cbar.set_label('Sigma (~Temperature)')
    temp_ax[k / 2][k % 2].set_title(pols[k])
    temp_ax[k / 2][k % 2].set_ylabel('Observation')
    temp_ax[k / 2][k % 2].set_xlabel('Frequency (Mhz)')
    temp_ax[k / 2][k % 2].set_xticks(xticks)
    temp_ax[k / 2][k % 2].set_xticklabels(xticklabels)
    temp_ax[k / 2][k % 2].set_aspect(float(sigma[pols[k]].shape[1]) / sigma[pols[k]].shape[0])
    temp_ax[k / 2][k % 2].xaxis.set_minor_locator(AutoMinorLocator(4))

plt.tight_layout()

hist_fig.savefig(outpath + 'HERA_Autos_Edges_Hist_Fit_All.png')
temp_fig.savefig(outpath + 'HERA_Autos_Edges_Temperatures_All.png')
