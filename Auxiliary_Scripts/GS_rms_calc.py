import numpy as np
import rfipy as rfi
from matplotlib import use, cm
use('Agg')
import matplotlib.pyplot as plt
import glob
import plot_lib as pl

rms_path = '/Users/mike_e_dubs/MWA/Temperatures/Golden_Set_RMS/arrs/'
outpath = '/Users/mike_e_dubs/MWA/Temperatures/Golden_Set_RMS/figs/'
rms_list = glob.glob('%s*.npym' % (rms_path))
rms_list.sort()

rms_arr = np.zeros([2, len(rms_list)])
xticks = [1061313496, 1061315320, 1061317152, 1061318984]

for k, rms in enumerate(rms_list):
    rms_arr[0, k] = int(rms[len(rms_path):len(rms_path) + 10])
    rms_arr[1, k] = np.load(rms)

fig, ax = plt.subplots(figsize=(14, 8))
pl.scatter_plot_2d(fig, ax, rms_arr[0], rms_arr[1], title='Golden Set RMS',
                   xlabel='Obsid', ylabel='RMS', xticks=xticks)
fig.savefig('%sGolden_Set_RMS.png' % (outpath))
