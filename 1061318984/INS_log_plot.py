from SSINS import INS, plot_lib, util
import numpy as np
import matplotlib.pyplot as plt

inpath = '/Users/mike_e_dubs/General/1061318984'
outpath = '%s/figs' % inpath
obs = '1061318984'
read_paths = util.read_paths_construct(inpath, None, obs, 'INS')

ins = INS(read_paths=read_paths, obs=obs, outpath=outpath)

vmax = [None, 150]
fig, ax = plt.subplots(figsize=(14, 8), nrows=2)
fig.suptitle('Narrowband SSINS')

for i in range(2):

    plot_lib.image_plot(fig, ax[i], ins.data[:, 0, :, 0],
                        cbar_label='Amplitude (UNCALIB)',
                        freq_array=ins.freq_array[0], vmax=vmax[i])
fig.savefig('%s/1061318984_None_INS_data_XX.png' % outpath)
