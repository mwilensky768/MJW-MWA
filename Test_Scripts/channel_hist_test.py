import rfiutil
import plot_lib as pl
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.stats
from scipy.special import erfc, erfinv
from math import pi
import glob
from matplotlib import cm

indir = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Golden_Set_8s_Autos/Vis_Avg/Averages'
# obsids = ['1131739792', '1131739912', '1131740032', '1131740152', '1131740272',
            # '1131740392', '1131740512', '1131740632', '1131740752']
obs_list = '/Users/mike_e_dubs/python_stuff/MJW-MWA/Obs_Lists/GS_NB.txt'
with open(obs_list) as f:
    obsids = f.readlines()
obsids = [obsid.strip() for obsid in obsids]
arrs = ['%s/%s_Vis_Avg_Amp_All.npy' % (indir, obsid) for obsid in obsids]
# arrs = glob.glob('%s/*.npy' % (indir))
arrs.sort()
outpath = '/Users/mike_e_dubs/MWA/Test_Plots/match_filter/GS_NB_Hists/'
freq_arr = np.zeros([1, 384])
freq_arr[0, :] = np.load('/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/MWA_Highband_Freq_Array.npy')
if not os.path.exists(outpath):
    os.makedirs(outpath)
C = (4 / np.pi - 1)

counts = []

for path in arrs:
    INS = np.ma.masked_array(np.load(path))
    Nbls = 8001 * np.ones(INS.shape)
    MS = (INS / INS.mean(axis=0) - 1) * np.sqrt(Nbls / C)
    obs = path[len(indir) + 1:len(indir) + 11]
    fig1, ax1 = plt.subplots(nrows=2, ncols=2)
    fig2, ax2 = plt.subplots(nrows=2, ncols=2)
    fig1.suptitle('RFI Contaminated')
    fig2.suptitle('Clean')
    for i in range(INS.shape[3]):
        hist1, bins1 = np.histogram(INS[:, 0, 162, i], bins='auto')
        hist2, bins2 = np.histogram(INS[:, 0, 180, i], bins='auto')
        pl.one_d_hist_plot(fig1, ax1[i / 2][i % 2], bins1, [hist1, ], xlog=False)
        pl.one_d_hist_plot(fig2, ax2[i / 2][i % 2], bins2, [hist2, ], xlog=False)
    fig1.savefig('%s%s_NB_hist.png' % (outpath, obs))
    fig2.savefig('%s%s_NB_hist_clean.png' % (outpath, obs))
    plt.close(fig1)
    plt.close(fig2)
