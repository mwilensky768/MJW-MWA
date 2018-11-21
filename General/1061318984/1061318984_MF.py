from SSINS import INS, MF, util, plot_lib
from SSINS import Catalog_Plot as cp
from matplotlib import cm
import matplotlib.pyplot as plt

indir = '/Users/mike_e_dubs/General/1061318984'
outdir = '%s_Filtered' % indir
obs = '1061318984'
read_paths = util.read_paths_construct(indir, None, obs, 'INS')

sig_thresh = 5

ins = INS(read_paths=read_paths, obs=obs, outpath=outdir)
mf = MF(ins, sig_thresh=sig_thresh, N_thresh=15)
mf.apply_match_test()
fig, ax = plt.subplots(figsize=(14, 8), nrows=3)
fig.suptitle('Narrowband SSINS')
plot_lib.image_plot(fig, ax[0], ins.data_ms[:, 0, :, 0], cmap=cm.coolwarm,
                    vmin=-5, vmax=5, title='Before N Sample Test',
                    cbar_label='Deviation ($\hat{\sigma}$)', mask_color='black',
                    freq_array=ins.freq_array[0])
ax[0].set_xlabel('')
plot_lib.image_plot(fig, ax[1], ins.data[:, 0, :, 0], cmap=cm.viridis,
                    mask_color='white', cbar_label='Deviation ($\hat{\sigma}$)',
                    freq_array=ins.freq_array[0])
ax[1].set_xlabel('')
ins.data.mask = False
ins.data_ms = ins.mean_subtract()
mf.apply_match_test(apply_N_thresh=True)
plot_lib.image_plot(fig, ax[2], ins.data[:, 0, :, 0], cmap=cm.viridis,
                    title='After N Sample Test', mask_color='white',
                    cbar_label='Deviation ($\hat{\sigma}$)',
                    freq_array=ins.freq_array[0])

fig.savefig('%s/figs/Before_Samp_Thresh.png' % outdir)
