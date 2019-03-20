from SSINS import INS, util, plot_lib, MF
from matplotlib import cm
import matplotlib.pyplot as plt

obs = '1093972368'
inpath = '/Users/mike_e_dubs/General/1093972368'
outpath = inpath

read_paths = util.read_paths_construct(inpath, None, obs, 'INS')
ins = INS(read_paths=read_paths, outpath=outpath, obs=obs)
aspect = float(ins.data.shape[2]) / ins.data.shape[0]

fig, ax = plt.subplots(figsize=(8, 9))
plot_lib.image_plot(fig, ax, ins.data[:, 0, :, 0], cbar_label='Amplitude (UNCALIB)',
                    freq_array=ins.freq_array[0], aspect=aspect)

fig.savefig('%s/%s_INS.png' % (outpath, obs))



fig_new, ax_new = plt.subplots(figsize=(16, 9), ncols=2)

plot_lib.image_plot(fig, ax_new[0], ins.data_ms[:, 0, :, 0], cbar_label='Deviation ($\hat{\sigma}$)',
                    freq_array=ins.freq_array[0], cmap=cm.coolwarm, aspect=aspect)

mf = MF(ins, sig_thresh=5)
mf.apply_match_test()

plot_lib.image_plot(fig_new, ax_new[1], ins.data_ms[:, 0, :, 0], cbar_label='Deviation ($\hat{\sigma}$)',
                    freq_array=ins.freq_array[0], cmap=cm.coolwarm, aspect=aspect,
                    mask_color='black')
fig_new.savefig('%s/%s_INS_MS.png' % (outpath, obs))
