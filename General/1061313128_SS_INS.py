from __future__ import division

from SSINS import SS, plot_lib, util, INS
from matplotlib import cm
import matplotlib.pyplot as plt
import os

obs = '1061313128'
inpath = '/Users/mike_e_dubs/MWA/Data/uvfits/%s_noflag.uvfits' % obs
outpath = '/Users/mike_e_dubs/General/%s' % obs

bad_time_indices = [0, -1, -2, -3]


if not os.path.exists('%s/arrs/%s_None_INS_data.npym' % (outpath, obs)):
    ss = SS(inpath=inpath, outpath=outpath, obs=obs, bad_time_indices=bad_time_indices,
            read_kwargs={'ant_str': 'cross'})
    ss.INS_prepare()
    ss.INS.save()
    ins = ss.INS
else:
    read_paths = util.read_paths_construct(outpath, None, obs, 'INS')
    ins = INS(read_paths=read_paths, obs=obs, outpath=outpath)

fig, ax = plt.subplots(figsize=(8, 9))
plot_lib.image_plot(fig, ax, ins.data[:, 0, :, 0], aspect=ins.data.shape[2] / ins.data.shape[0],
                    freq_array=ins.freq_array[0], cbar_label='Amplitude (UNCALIB)')
fig.savefig('%s/%s_INS.png' % (outpath, obs))
plt.close(fig)
fig, ax = plt.subplots(figsize=(8, 9))
plot_lib.image_plot(fig, ax, ins.data_ms[:, 0, :, 0], aspect=ins.data.shape[2] / ins.data.shape[0],
                    freq_array=ins.freq_array[0], cbar_label='Amplitude (UNCALIB)',
                    cmap=cm.coolwarm)
fig.savefig('%s/%s_INS_ms.png' % (outpath, obs))
