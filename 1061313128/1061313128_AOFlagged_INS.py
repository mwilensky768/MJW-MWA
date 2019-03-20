from __future__ import division

from SSINS import SS, plot_lib, util, INS
import matplotlib.pyplot as plt
from matplotlib import cm
import os

obs = '1061313128'
inpath = '/Users/mike_e_dubs/MWA/Data/uvfits/1061313128.uvfits'
inpath2 = '/Users/mike_e_dubs/MWA/Data/uvfits/1061313128_noflag.uvfits'
outpath = '/Users/mike_e_dubs/General/1061313128'

if not os.path.exists('%s/arrs/%s_original_INS_data.npym' % (outpath, obs)):
    ss = SS(obs=obs, inpath=inpath, bad_time_indices=[0, -1, -2, -3],
            read_kwargs={'ant_str': 'cross'}, flag_choice='original',
            outpath=outpath)
    ss.INS_prepare()
    ss.INS.save()
    ins = ss.INS
else:
    read_paths = util.read_paths_construct(outpath, 'original', obs, 'INS')
    ins = INS(read_paths=read_paths, outpath=outpath, obs=obs, flag_choice='original')
fig, ax = plt.subplots(figsize=(8, 9))
plot_lib.image_plot(fig, ax, ins.data_ms[:, 0, :, 0], cmap=cm.coolwarm,
                    cbar_label='Deviation ($\hat{\sigma}$)', aspect=ins.data.shape[2] / ins.data.shape[0],
                    freq_array=ins.freq_array[0], mask_color='black')
fig.savefig('%s/1061313128_AOFlagger_INS_ms.png' % outpath)
