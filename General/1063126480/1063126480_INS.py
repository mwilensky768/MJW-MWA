from SSINS import INS, MF, plot_lib, util
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

obs = '1063126480'
indir = '/Users/mike_e_dubs/MWA/INS/Long_Run/All'
outpath = '/Users/mike_e_dubs/General/%s' % obs

read_paths = util.read_paths_construct(indir, None, obs, 'INS')
ins = INS(read_paths=read_paths, obs=obs, outpath=outpath)
mf = MF(ins, sig_thresh=5)

fig, ax = plt.subplots(figsize=(14, 8), nrows=2)
plot_lib.image_plot(fig, ax[0], ins.data_ms[:, 0, :, 0], cmap=cm.coolwarm,
                    freq_array=ins.freq_array[0], title='MWA Soft Breather',
                    cbar_label='Deviation ($\hat{\sigma}$)')

mf.apply_match_test()
plot_lib.image_plot(fig, ax[1], ins.data_ms[:, 0, :, 0], cmap=cm.coolwarm, vmin=-5,
                    vmax=5, freq_array=ins.freq_array[0], title='MWA Soft Breather',
                    mask_color='black', cbar_label='Deviation ($\hat{\sigma}$)')
fig.savefig('%s/figs/%s_Order_0.png' % (outpath, obs))
fig2, ax2 = plt.subplots(figsize=(14, 8))
ins.data.mask = False
ins.data_ms = ins.mean_subtract(order=1)
mf.apply_match_test(order=1)
plot_lib.image_plot(fig2, ax2, ins.data_ms[:, 0, :, 0], cmap=cm.coolwarm, vmin=-5,
                    vmax=5, freq_array=ins.freq_array[0], title='MWA Soft Breather (Linear Correction)',
                    mask_color='black', cbar_label='Deviation ($\hat{\sigma}$)')
fig2.savefig('%s/figs/%s_Order_1.png' % (outpath, obs))
