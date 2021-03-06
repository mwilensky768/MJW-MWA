from pyuvdata import UVData
from SSINS import plot_lib
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('obs')
parser.add_argument('inpath')
parser.add_argument('outpath')
args = parser.parse_args()
if not os.path.exists(args.outpath):
    os.makedirs(args.outpath)

UV = UVData()
UV.read(args.inpath, polarizations=-5)
UV.select(ant_str='cross')
UV.data_array = UV.data_array.reshape([UV.Ntimes, UV.Nbls, UV.Nspws, UV.Nfreqs,
                                       UV.Npols])
UV.data_array = UV.data_array[1:-3]

ylabel = ['Time (2s)', 'Time Pair']
tag = ['vis', 'diff']
title = ['MWA Single Baseline Visibilities', 'MWA Single Baseline Visibility Differences']
for i in range(UV.Nbls):
    im_dat = UV.data_array[:, i, 0, :, 0]
    for k in range(2):
        title = 'MWA Single Baseline Visibilities'
        if k:
            im_dat = np.diff(im_dat, axis=0)
            title = 'MWA Single Baseline Visibility Differences'
        for label in ['Real', 'Imag']:
            fig, ax = plt.subplots(figsize=(14, 8))
            if getattr(im_dat, label.lower()).size:
                plot_lib.image_plot(fig, ax, getattr(im_dat, label.lower()), ylabel=ylabel[k],
                                    cmap=cm.RdGy_r,
                                    title='%s (%s)' % (title, label),
                                    freq_array=UV.freq_array[0], cbar_label=UV.vis_units)
                fig.savefig('%s/%s_%i_%s_%s.png' % (args.outpath, args.obs, i, tag[k], label))
                plt.close(fig)
            else:
                print('baseline %i was empty for %s attr' % (i, label))
        if im_dat.size:
            fig, ax = plt.subplots(figsize=(14, 8))
            plot_lib.image_plot(fig, ax, np.absolute(im_dat), cbar_label=UV.vis_units,
                                title='%s (Amplitude)' % title,
                                freq_array=UV.freq_array[0], ylabel=ylabel[k])
            fig.savefig('%s/%s_%i_%s_amplitude.png' % (args.outpath, args.obs, i, tag[k]))
            plt.close(fig)
        else:
            print('im_dat was empty for baseline %i', i)
