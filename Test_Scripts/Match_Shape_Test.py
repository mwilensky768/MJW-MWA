import rfiutil
import numpy as np
import plot_lib as pl
from matplotlib import cm
from matplotlib.ticker import AutoMinorLocator
import glob
import matplotlib.pyplot as plt
import os
from time import strftime

arr_path = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Golden_Set_8s_Autos/Vis_Avg/Averages/'
outpath = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Templates/Golden_Set_8s_Autos/Match_Filter/'
arr_list = glob.glob('%s*.npy' % (arr_path))
arr_list.sort()
occ_num = np.zeros(22)
occ_den = np.copy(occ_num)
occ_freq_num = np.zeros([22, 384])
occ_freq_den = np.zeros(22)
freq_array = np.zeros([1, 384])
freq_array[0, :] = np.load('/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/MWA_Highband_Freq_Array.npy')
centers = [8 + 16 * k for k in range(24)]
ch_ignore = centers
shape_dict = {'TV%i' % (k): np.load('/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/TV%i_freqs.npy' % (k))
              for k in [6, 7, 8]}

for arr in arr_list:
    obs = arr[len(arr_path):len(arr_path) + 10]
    print('%s started at %s' % (obs, strftime("%H:%M:%S")))
    for sig_thresh in range(4, 5):
        if sig_thresh % 2 == 0:
            print('%i started at %s' % (sig_thresh, strftime("%H:%M:%S")))
        INS = np.ma.masked_array(np.load(arr))
        # INS = rfiutil.narrowband_filter(INS, ch_ignore)
        MS = INS / INS.mean(axis=0) - 1
        if sig_thresh == 4:
            Nbls = 8001 * np.ones(INS.shape)

        INS, MS, _, _, _, _ = rfiutil.match_filter(INS, MS, Nbls, freq_array, sig_thresh, shape_dict)
        occ_num[sig_thresh - 4] += np.count_nonzero(INS.mask)
        occ_den[sig_thresh - 4] += np.prod(INS.shape)
        if np.count_nonzero(INS.mask) > 0:
            occ_freq_num[sig_thresh - 4, :] += np.count_nonzero(INS.mask, axis=(0, 1, 3))
        occ_freq_den[sig_thresh - 4] += INS.shape[0] * INS.shape[3]

        fig, ax, pols, xticks, xminors, yminors, xticklabels = pl.four_panel_tf_setup(freq_array[0, :])
        for m in range(4):
            pl.image_plot(fig, ax[m / 2][m % 2], MS[:, 0, :, m], cmap=cm.coolwarm,
                          title=pols[m], xlabel='Frequency (Mhz)', ylabel='Time Pair',
                          cbar_label='Fraction of Mean', xticks=xticks, xminors=xminors,
                          yminors=yminors, xticklabels=xticklabels, zero_mask=False,
                          mask_color='black')

        if not os.path.exists('%s%i/' % (outpath, sig_thresh)):
            os.makedirs('%s%i/' % (outpath, sig_thresh))
        fig.savefig('%s%i/%s_match_filter_MS.png' % (outpath, sig_thresh, obs))
        plt.close(fig)

occ = occ_num / occ_den * 100
occ_freq = (occ_freq_num.transpose() / occ_freq_den).transpose() * 100
np.save('%sRFI_occupancy_sigma.npy' % (outpath), occ)
np.save('%sRFI_occupancy_freq.npy' % (outpath), occ_freq)
fig, ax = plt.subplots(figsize=(14, 8))
fig_f, ax_f = plt.subplots(figsize=(14, 8))
xticks = range(-5, 25, 5)
xticklabels = [str(tick + 4) for tick in xticks]
pl.line_plot(fig, ax, [occ, ], title='RFI Occupancy at Different Thresholds',
             xlabel='Sigma', ylabel='Percent Occupancy', xticks=xticks,
             xticklabels=xticklabels, legend=False)
xticks = [64 * k for k in range(6)]
xminors = AutoMinorLocator(4)
xticklabels = ['%.1f' % (freq_array[0, tick] * 10 ** (-6)) for tick in xticks]
pl.line_plot(fig_f, ax_f, occ_freq,
             title='RFI Occupancy at Different Frequencies', xlabel='Frequency (Mhz)',
             ylabel='Percent Occupancy', xticks=xticks, xminors=xminors,
             labels=['%i$\sigma$' % (k) for k in range(4, 26)],
             xticklabels=xticklabels, legend=True)
fig.savefig('%sRFI_Occupany.png' % (outpath))
fig_f.savefig('%sRFI_Occupancy_Freqs.png' % (outpath))
