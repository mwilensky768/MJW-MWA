from __future__ import division

from SSINS import INS, plot_lib, util
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import cm

obs = '1066742016'
indir = '/Users/mike_e_dubs/MWA/INS/Long_Run/All'
outpath = '/Users/mike_e_dubs/General/1066742016'
if not os.path.exists(outpath):
    os.makedirs(outpath)

read_paths = util.read_paths_construct(indir, None, obs, 'INS')
ins = INS(read_paths=read_paths, outpath=outpath, obs=obs)
fig, ax = plt.subplots(figsize=(8 / 3, 3))
fig_diff, ax_diff = plt.subplots(figsize=(8 / 3, 3))

plot_lib.image_plot(fig, ax, ins.data[:, 0, :, 0], freq_array=ins.freq_array[0],
                    aspect=ins.data.shape[2] / ins.data.shape[0], cbar_label='Amplitude (arbs)',)
plot_lib.image_plot(fig_diff, ax_diff, ins.data_ms[:, 0, :, 0],
                    freq_array=ins.freq_array[0], cbar_label='Deviation ($\hat{\sigma}$)',
                    aspect=ins.data.shape[2] / ins.data.shape[0], cmap=cm.coolwarm)
fig.savefig('%s/1066742016_ins.png' % outpath)
fig_diff.savefig('%s/1066742016_ms_nopin.png' % outpath)
