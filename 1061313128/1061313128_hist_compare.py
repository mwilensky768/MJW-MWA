from SSINS import SS, plot_lib
from pyuvdata import UVData
from matplotlib import pyplot as plt
import numpy as np

fig, ax = plt.subplots(figsize=(14, 8), nrows=2)
outpath = '/Users/mike_e_dubs/MWA/1061313128/Compare'
leg_labels = ['AOFlagger Applied', 'No AOFlagger']
titles = ['Visibilities', 'Visibility Differences']

for i, string in enumerate(['', '_noflag']):
    inpath = '/Users/mike_e_dubs/MWA/Data/uvfits/1061313128%s.uvfits' % string
    UV = UVData()
    UV.read(inpath, file_type='uvfits', read_data=False)
    freqs = np.logical_and(1.79e8 < UV.freq_array[0], 1.9e8 > UV.freq_array[0])
    UV.read(inpath, file_type='uvfits', read_metadata=False, freq_chans=freqs,
            ant_str='cross', times=np.unique(UV.time_array)[1:-3])
    for k in range(2):
        if k:
            ss = SS(UV=UV)
        n, bins = np.histogram(np.absolute(UV.data_array), bins='auto')
        cent = bins[:-1] + 0.5 * np.diff(bins)
        plot_lib.error_plot(fig, ax[k], cent, n, title=titles[k], label=leg_labels[i],
                            xlabel='Amplitude (UNCALIB)', ylabel='counts', yscale='log')
    del ss
    del UV

fig.savefig('%s/vis_hists.png' % outpath)
