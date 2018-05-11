import numpy as np
from matplotlib import use, cm
import plot_lib as pl
import matplotlib.pyplot as plt
import os

arr_path = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Golden_Set_8s_Autos/Vis_Avg/Averages/'
obs_list = [1061318616, 1061318736, 1061318864, 1061318984, 1061319104, 1061319224, 1061319352, 1061319472, 1061312392]
outpath = '/Users/mike_e_dubs/MWA/Test_Plots/Narrowband_Test/'
freq_path = '/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/MWA_Highband_Freq_Array.npy'
freq_array = np.load(freq_path)
if not os.path.exists(outpath):
    os.makedirs(outpath)
centers = [8 + 16 * k for k in range(24)]
LEdges = [0 + 16 * k for k in range(24)]
REdges = [15 + 16 * k for k in range(24)]
for obs in obs_list:
    arr = '%s%i_Vis_Avg_Amp_All.npy' % (arr_path, obs)
    INS = np.ma.masked_array(np.load(arr))
    FD_CB = np.zeros(INS.shape)
    for m in range(24):
        INS_CBM = np.array([np.median(INS[:, :, 16 * m:16 * (m + 1), :], axis=2) for k in range(16)]).transpose((1, 2, 0, 3))
        FD_CB[:, :, 16 * m:16 * (m + 1), :] = INS[:, :, 16 * m:16 * (m + 1), :] / INS_CBM - 1

    INS[:, :, centers, :] = np.ma.masked
    INS[:, :, LEdges, :] = np.ma.masked
    INS[:, :, REdges, :] = np.ma.masked

    FD_CB = np.ma.masked_where(np.logical_and(FD_CB > 0.1, ~INS.mask), FD_CB)
    if np.any(FD_CB.mask):
        FD_CB.mask[:, :, centers, :] = False
        FD_CB.mask[:, :, LEdges, :] = False
        FD_CB.mask[:, :, REdges, :] = False
    if obs == 1061312392:
        FD_CB[:, :, centers, :] = np.ma.masked
        FD_CB[:, :, LEdges, :] = np.ma.masked
        FD_CB[:, :, REdges, :] = np.ma.masked

    fig, ax, pols, xticks, xminors, yminors, xticklabels = pl.four_panel_tf_setup(freq_array)
    for m in range(4):
        pl.image_plot(fig, ax[m / 2][m % 2], FD_CB[:, 0, :, m], cmap=cm.coolwarm,
                      title=pols[m], cbar_label='Fraction', xticks=xticks,
                      xminors=xminors, yminors=yminors, xticklabels=xticklabels,
                      zero_mask=False, mask_color='black')
    fig.savefig('%s%i_Narrowband_FD_CB.png' % (outpath, obs))
    plt.close(fig)
