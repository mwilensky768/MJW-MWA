from SSINS import INS, util, plot_lib, MF
from matplotlib import cm
import matplotlib.pyplot as plt

obs = '1063126480'
inpath = '/Users/mike_e_dubs/MWA/INS/Long_Run/All'
outpath = '/Users/mike_e_dubs/General/%s' % obs
read_paths = util.read_paths_construct(inpath, None, obs, 'INS')
ins = INS(read_paths=read_paths, obs=obs, outpath=outpath)
aspect = float(ins.data.shape[2]) / ins.data.shape[0]

fig, ax = plt.subplots(figsize=(16, 9), ncols=2)
plot_lib.image_plot(fig, ax[0], ins.data_ms[:, 0, :, 0], cmap=cm.coolwarm,
                    freq_array=ins.freq_array[0], cbar_label='Deviation ($\hat{\sigma}$)',
                    aspect=aspect)
ins.data_ms = ins.mean_subtract(order=1)
plot_lib.image_plot(fig, ax[1], ins.data_ms[:, 0, :, 0], cmap=cm.coolwarm,
                    freq_array=ins.freq_array[0], cbar_label='Deviation ($\hat{\sigma}$)',
                    mask_color='black', aspect=aspect)
fig.savefig('%s/%s_INS_MS_Soft_Breather_order_1.png' % (outpath, obs))
