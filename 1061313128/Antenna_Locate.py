import numpy as np
from SSINS import plot_lib
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.io import readsav

ant_locs = np.load('/Users/mike_e_dubs/Repositories/MJW-MWA/Useful_Information/MWA_ant_pos.npy')
sav = readsav('/Users/mike_e_dubs/MWA/FHD/1061313128_Noflag/1061313128_cal.sav')

c = 128 * ['']
c = np.array(c)
print(np.absolute(sav.cal.gain[0][0][:, 6]) < 1)
c[np.absolute(sav.cal.gain[0][0][:, 6]) < 1] = 'r'
c[np.absolute(sav.cal.gain[0][0][:, 6]) > 1] = 'b'
fig, ax = plt.subplots(figsize=(14, 8))
fig_line, ax_line = plt.subplots(figsize=(14, 8))
plot_lib.scatter_plot_2d(fig, ax, ant_locs[:, 0], ant_locs[:, 1], c=c, xlabel='X (m)',
                         ylabel='Y (m)')
plot_lib.error_plot(fig_line, ax_line, range(len(ant_locs)), np.sqrt(ant_locs[:, 0]**2 + ant_locs[:, 1]**2),
                    ylabel='Radial Distance (m)', xlabel='Antenna Index')

fig.savefig('/Users/mike_e_dubs/MWA/FHD/1061313128_Noflag/ant_locs.png')
fig_line.savefig('/Users/mike_e_dubs/MWA/FHD/1061313128_Noflag/ant_rad_distance.png')
