import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pyuvdata
from math import floor, ceil, log10
from matplotlib.ticker import AutoMinorLocator

obslist_path = '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Long_Run_8s_Autos_OBSIDS.txt'
cutlist_path = '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Long_Run_8s_Autos_Funky_OBSIDS.txt'
inpath = '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Temperatures_All/'
freq_array_obs_path = '/Users/mike_e_dubs/python_stuff/smaller_uvfits/s1061313008.uvfits'
outpath = '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/'
corr_freq_chan = 128
bftemp_path = '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Long_Run_Avg_Obs_Temp.npy'
obs_count_path = '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Long_Run_Temp_Obs_Count.npy'

UV = pyuvdata.UVData()
UV.read_uvfits(freq_array_obs_path)

with open(obslist_path) as f:
    obslist = f.read().split("\n")
with open(cutlist_path) as h:
    cutlist = h.read().split("\n")

for item in cutlist:
    obslist.remove(item)

obslist = np.sort(np.array(obslist).astype(int))
obslist = obslist.astype(str)

pols = ['XX', 'YY', 'XY', 'YX']

n = np.load(inpath + obslist[0] + '_hist.npy')
bins = np.load(inpath + obslist[0] + '_bins.npy')
fit = np.load(inpath + obslist[0] + '_fit.npy')
sigma = {}
for pol in pols:
    sigma[pol] = np.zeros([len(obslist), UV.Nfreqs])
    sigma[pol][0, :] = np.load(inpath + obslist[0] + '_sigma_' + pol + '.npy')

k = 1
for obs in obslist[1:]:
    n += np.load(inpath + obs + '_hist.npy')
    fit += np.load(inpath + obs + '_fit.npy')
    for pol in pols:
        sigma[pol][k, :] = np.load(inpath + obs + '_sigma_' + pol + '.npy')
    k += 1

residual = n - fit

widths = np.diff(bins)
centers = bins[:-1] + 0.5 * widths
N = len(n)

corr_sigma = 0.25 * (sigma['XX'][:, corr_freq_chan] + sigma['YY'][:, corr_freq_chan] +
                     sigma['XY'][:, corr_freq_chan] + sigma['YX'][:, corr_freq_chan])
corr_temp = np.load(bftemp_path) + 273.15 * np.ones(len(corr_sigma))
corr_obs_count = np.load(obs_count_path)

hist_fig, hist_ax = plt.subplots(figsize=(14, 8), nrows=2)
hist_ax[0].step(bins[:-1], n, where='pre', label='Histogram (No Flag Cut)')
hist_ax[0].plot(centers, fit, label='Fit')
hist_ax[0].set_title('Visibility Difference Histogram, Long Run, 8s/Autos Removed')
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
auto_pol_max = max([np.amax(sigma['XX']), np.amax(sigma['YY'])])
cross_pol_max = max([np.amax(sigma['XY']), np.amax(sigma['YX'])])
auto_pol_min = min([np.amin(sigma['XX']), np.amin(sigma['YY'])])
cross_pol_min = min([np.amin(sigma['XY']), np.amin(sigma['YX'])])
vmax = dict(zip(pols, [auto_pol_max, auto_pol_max, cross_pol_max, cross_pol_max]))
vmin = dict(zip(pols, [auto_pol_min, auto_pol_min, cross_pol_min, cross_pol_min]))
xticks = [UV.Nfreqs / 6 * l for l in range(6)]
xticks.append(UV.Nfreqs - 1)
xticklabels = ['%.1f' % (UV.freq_array[0, k] * 10**(-6)) for k in xticks]
for k in range(4):
    sigma[pols[k]] = np.ma.masked_equal(sigma[pols[k]], 0)
    cmap = cm.cool
    cmap.set_bad(color='white')
    cax = temp_ax[k / 2][k % 2].imshow(sigma[pols[k]], cmap=cmap, vmin=vmin[pols[k]], vmax=vmax[pols[k]])
    cbar = temp_fig.colorbar(cax, ax=temp_ax[k / 2][k % 2])
    cbar.set_label('Sigma (~Temperature)')
    temp_ax[k / 2][k % 2].set_title(pols[k])
    temp_ax[k / 2][k % 2].set_ylabel('Observation')
    temp_ax[k / 2][k % 2].set_xlabel('Frequency (Mhz)')
    temp_ax[k / 2][k % 2].set_xticks(xticks)
    temp_ax[k / 2][k % 2].set_xticklabels(xticklabels)
    temp_ax[k / 2][k % 2].set_aspect(float(sigma[pols[k]].shape[1]) / sigma[pols[k]].shape[0])
    temp_ax[k / 2][k % 2].xaxis.set_minor_locator(AutoMinorLocator(4))


corr_fig, corr_ax = plt.subplots(figsize=(14, 8))
corr_ax.set_title('Ambient Temperature vs. Sigma f = %.1f Mhz (MWA)' %
                  (UV.freq_array[0, corr_freq_chan] * 10**(-6)) + ' Mhz (MWA)')
corr_ax.set_xlabel('Ambient Temperature (K)')
corr_ax.set_ylabel('Sigma')
corr_ax.scatter(corr_temp[corr_obs_count > 0], corr_sigma[corr_obs_count > 0])

plt.tight_layout()

hist_fig.savefig(outpath + 'Long_Run_8s_Autos_Hist_Fit_All.png')
temp_fig.savefig(outpath + 'Long_Run_8s_Autos_Temperatures_All.png')
corr_fig.savefig(outpath + 'Long_Run_8s_Autos_Temperature_Correlation.png')
