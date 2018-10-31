import numpy as np
from scipy.io import readsav
from SSINS import plot_lib
import matplotlib.pyplot as plt
from matplotlib import cm

indir = '/Users/mike_e_dubs/MWA/FHD/1061313128_Noflag'
noflag = readsav('%s/1061313128_cal.sav' % indir)
cotter = readsav('%s/1061313128_COTTER_cal.sav' % indir)
ssins = readsav('%s/1061313128_SSINS_cal.sav' % indir)

pols = ['X', 'Y']
funcs = ['absolute', 'angle']
titles = ['Amplitude', 'Phase']
colormaps = [cm.viridis, cm.hsv]
res_colormap = [cm.coolwarm, cm.hsv]

for i in range(2):
    fig, ax = plt.subplots(figsize=(14, 8), nrows=3, ncols=2)
    for k in range(2):
        plot_lib.image_plot(fig, ax[0, k],
                            getattr(np, funcs[i])(noflag.cal.gain[0][k]),
                            xlabel='Channel #', ylabel='Antenna', cbar_label='No Flagging',
                            title='%s %s' % (pols[k], titles[i]), aspect=3, cmap=colormaps[i])
        plot_lib.image_plot(fig, ax[2, k],
                            getattr(np, funcs[i])(noflag.cal.gain[0][k]) - getattr(np, funcs[i])(ssins.cal.gain[0][k]),
                            xlabel='Channel #', ylabel='Antenna', cbar_label='Residual',
                            title='%s %s' % (pols[k], titles[i]), aspect=3, cmap=res_colormap[i])
        plot_lib.image_plot(fig, ax[1, k],
                            getattr(np, funcs[i])(ssins.cal.gain[0][k]),
                            xlabel='Channel #', ylabel='Antenna', cbar_label='SSINS',
                            title='%s %s' % (pols[k], titles[i]), aspect=3, cmap=colormaps[i])
    fig.savefig('%s/%s_Compare_phase_deref.png' % (indir, titles[i]))
    plt.close(fig)
