import numpy as np
import plot_lib as pl
import matplotlib.pyplot as plt
from matplotlib import cm

ant_pos_arr = np.load('/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/MWA_ant_pos.npy')

fig, ax = plt.subplots(figsize=(14, 8))
pl.scatter_plot_2d(fig, ax, ant_pos_arr[:, 0], ant_pos_arr[:, 1],
                   title='Antennas by Color', xlabel='X (m)', ylabel='Y (m)',
                   c=np.arange(128).astype(float), cmap=cm.plasma)

fig.savefig('/Users/mike_e_dubs/MWA/Test_Plots/MWA_Ant_by_color.png')
