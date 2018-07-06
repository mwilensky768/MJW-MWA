import numpy as np
import plot_lib as pl
import matplotlib.pyplot as plt
import rfiutil
import glob
import os

indir = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Golden_Set_8s_Autos/Vis_Avg/Averages'
arrs = glob.glob('%s/*.npy' % (indir))
arrs.sort()
shape_dict = {'TV%i' % (k): np.load('/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/TV%i_freqs.npy' % (k))
              for k in [6, 7, 8]}
freq_array = np.zeros([1, 384])
freq_array[0, :] = np.load('/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/MWA_Highband_Freq_Array.npy')
outpath = '/Users/mike_e_dubs/MWA/Test_Plots/Match_Test_Samp_Thresh'
C = (4 / np.pi - 1)
pols = ['XX', 'YY', 'XY', 'YX']


for arr in arrs:
    obs = arr[len(indir) + 1:len(indir) + 11]
    for samp_thresh in range(10, 55, 5):
        INS = np.ma.masked_array(np.load(arr))
        Nbls = 8001 * np.ones(INS.shape)
        MS = (INS / INS.mean(axis=0) - 1) * np.sqrt(Nbls / C)
        INS, MS, events, hists = rfiutil.match_filter(INS, MS, Nbls, outpath, freq_array, sig_thresh=4,
                                                      shape_dict=shape_dict, samp_thresh=samp_thresh)
        fig, ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
        for i in range(4):
            pl.image_plot(fig, ax[i / 2][i % 2], INS[:, 0, :, i], title=pols[i],
                          cbar_label='Amplitude (UNCALIB)')
        if not os.path.exists('%s/%i' % (outpath, samp_thresh)):
            os.makedirs('%s/%i' % (outpath, samp_thresh))
        fig.savefig('%s/%i/%s_INS_MF_ST%i.png' % (outpath, samp_thresh, obs, samp_thresh))
        plt.close(fig)
