from __future__ import division

from SSINS import INS
from SSINS import util
from SSINS import MF
from SSINS import plot_lib as pl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os

# 'TV7_ext': [1.845e8 - 5.15e6, 1.845e8 + 5.15e6]
basedir = '/Users/mike_e_dubs/MWA/INS/Long_Run/All'
obs = '1066742016'
outpath = '/Users/mike_e_dubs/General/Movie'
if not os.path.exists(outpath):
    os.makedirs(outpath)
flag_choice = 'None'
read_paths = util.read_paths_construct(basedir, flag_choice, obs, 'INS')
shape_dict = {'TV6': [1.74e8, 1.81e8],
              'TV7': [1.81e8, 1.88e8],
              'TV8': [1.88e8, 1.95e8]}
order = 0
ins = INS(obs=obs, read_paths=read_paths, outpath=outpath, order=order)
mf = MF(ins, shape_dict=shape_dict, sig_thresh=5)
mf.apply_match_test(order=order)
ins.data.mask = False
ins.data_ms = ins.mean_subtract(order=order)
ins.outpath = '%s_0' % outpath
labels = ['', '_ms']
titles = ['', ' (Mean-Subtracted)']
cbar_labels = [ins.vis_units, 'Deviation ($\hat{\sigma}$)']
cmaps = [cm.viridis, cm.coolwarm]
for i in range(2):
    fig, ax = plt.subplots(figsize=(16, 9))
    if i:
        vmin = -4
        vmax = 4
    else:
        vmin = None
        vmax = None
    pl.image_plot(fig, ax, getattr(ins, 'data%s' % labels[i])[:, 0, :, 0],
                  xlabel='Frequency (Mhz)',
                  title='MWA Incoherent Noise Spectrum %s' % titles[i],
                  freq_array=ins.freq_array[0], cbar_label=cbar_labels[i],
                  cmap=cmaps[i], aspect=ins.data_ms.shape[2] / ins.data_ms.shape[0])
    fig.savefig('%s/%s_INS%s_master.png' % (outpath, obs, labels[i]))
    plt.close(fig)
for i, event in enumerate(ins.match_events):
    ins.outpath = '%s_%i' % (outpath, i + 1)
    ins.data[tuple(event)] = np.ma.masked
    ins.data_ms = ins.mean_subtract(order=order)
    fig, ax = plt.subplots(figsize=(16, 9))
    pl.image_plot(fig, ax, ins.data_ms[:, 0, :, 0],
                  xlabel='Frequency (Mhz)',
                  title='MWA Incoherent Noise Spectrum (Mean-Subtracted)',
                  freq_array=ins.freq_array[0], cbar_label=cbar_labels[1],
                  cmap=cm.coolwarm, mask_color='black',
                  aspect=ins.data_ms.shape[2] / ins.data_ms.shape[0])
    fig.savefig('%s/%s_INS_ms_%i.png' % (outpath, obs, i))
    plt.close(fig)
