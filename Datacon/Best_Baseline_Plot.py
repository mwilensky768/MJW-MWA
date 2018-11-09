from __future__ import division

import numpy as np
from pyuvdata import UVData
from SSINS import plot_lib
from matplotlib import cm
import matplotlib.pyplot as plt
import os

inpath = '/Volumes/Faramir/uvfits/1066742016.uvfits'
outpath = '/Users/mikewilensky/General'
i = 7380
freq_array = np.load('/Users/mikewilensky/Repos/MJW-MWA/Useful_Information/MWA_Highband_Freq_Array.npy')

if not os.path.exists(outpath):
    os.makedirs(outpath)

if os.path.exists('%s/Best_Baseline.npy' % outpath):
    data = np.load('%s/Best_Baseline.npy' % outpath)
else:
    UV = UVData()
    UV.read(inpath, file_type='uvfits', polarizations=-5)
    UV.select(ant_str='cross', times=np.unique(UV.time_array)[1:-3])
    UV.data_array = UV.data_array.reshape([UV.Ntimes, UV.Nbls, UV.Nspws, UV.Nfreqs, UV.Npols])
    data = UV.data_array[:, i, 0, :, 0]

data_diff = np.absolute(np.diff(data, axis=0))
fig, ax = plt.subplots(figsize=(8, 9))
fig_diff, ax_diff = plt.subplots(figsize=(8, 9))
np.save('%s/Best_Baseline.npy' % outpath, data)
print(data.shape[1] / (data.shape[0] * 10))
print(data.shape)
plot_lib.image_plot(fig, ax, data.real, cmap=cm.RdGy_r, freq_array=freq_array,
                    cbar_label='Amplitude (UNCALIB)', aspect=data.shape[1] / (data.shape[0]),
                    ylabel='Time (2s)')
plot_lib.image_plot(fig_diff, ax_diff, data_diff, freq_array=freq_array, ylabel='Time (2s)'
                    cbar_label='Amplitude (UNCALIB)', aspect=data.shape[1] / (data.shape[0]))
fig.savefig('%s/Best_Baseline.png' % outpath)
fig_diff.savefig('%s/Best_Baseline_Diff.png' % outpath)
