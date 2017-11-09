import pyuvdata
import numpy as np
from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt
import glob
import os

arr_path_list = glob.glob('/nfs/eor-00/h1/mwilensk/S2_Zenith_Calcut_8s_Autos/Catalogs/Ant_Pol/Chirp_Arr/*.npy')
UV = pyuvdata.UVData()
UV.read_uvfits('/nfs/eor-10/r1/EoRuvfits/jd2456855v5_1/1089578304/1089578304.uvfits')
outpath = '/nfs/eor-00/h1/mwilensk/S2_Zenith_Calcut_8s_Autos/Catalogs/Ant_Scatter/Chirp/'
c = UV.Nants_telescope * ['b']
pols = ['XX', 'YY', 'XY', 'YX']

for path in arr_path_list:
    indices = np.load(path)
    obs = path[-18:-8]

    for k in range(len(indices[0])):
        c = UV.Nants_telescope * ['b']
        if not os.path.exists('%s%s_ant_scatter_bl%i_pol%i_f%i_t%i.png' %
                              (outpath, obs, indices[1][k], indices[4][k], indices[3][k],
                               indices[0][k])):

            c[UV.ant_1_array[indices[1][k]]] = 'r'
            c[UV.ant_2_array[indices[1][k]]] = 'r'
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.scatter(UV.antenna_positions[:, 0], UV.antenna_positions[:, 1], c=c)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title('%s Antenna Lightup %s t = %i f = %.1f Mhz' %
                         (obs, pols[indices[4][k]], indices[0][k],
                          UV.freq_array[0, indices[3][k]] * 10 ** (-6)))
            fig.savefig('%s%s_ant_scatter_bl%i_pol%i_f%i_t%i.png' %
                        (outpath, obs, k, indices[4][k], indices[3][k],
                         indices[0][k]))
            plt.close(fig)
