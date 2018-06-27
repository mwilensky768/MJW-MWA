import rfiutil
import numpy as np
import matplotlib.pyplot as plt
import plot_lib as pl
import os
from matplotlib.ticker import AutoMinorLocator

obsids = [1061313128, 1061313008, 1061318984, 1061318864]
indir = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Golden_Set_8s_Autos/Vis_Avg/Averages'
arr_tag = '_Vis_Avg_Amp_All.npy'
freq_arr_path = '/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/MWA_Highband_Freq_Array.npy'
freqs = np.zeros([1, 384])
freqs[0, :] = np.load(freq_arr_path)
outpath = '/Users/mike_e_dubs/MWA/Test_Plots/match_filter_hist_converge_4sig'
if not os.path.exists(outpath):
    os.makedirs(outpath)
shape_dir = '/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information'

shape_dict = {'TV%i' % (i): np.load('%s/TV%i_freqs.npy' % (shape_dir, i)) for i in [6, 7, 8]}
pols = ['XX', 'YY', 'XY', 'YX']
C = 4 / np.pi - 1
image_plot_kwargs = {'xticks': [64 * k for k in range(6)],
                     'xticklabels': [freqs[0, tick] * 10 ** (-6) for tick in [64 * k for k in range(6)]],
                     'xminors': AutoMinorLocator(4),
                     'cbar_label': 'Amplitude (UNCALIB)',
                     'xlabel': 'Frequency (Mhz)'}

for obs in obsids:
    INS = np.ma.masked_array(np.load('%s/%s%s' % (indir, obs, arr_tag)))
    Nbls = 8001 * np.ones(INS.shape)
    MS = (INS / INS.mean(axis=0) - 1) * np.sqrt(Nbls / C)

    INS, MS, n, bins, fit, events, hists = \
        rfiutil.match_filter(INS, MS, Nbls, outpath, freqs, shape_dict=shape_dict)

    if not os.path.exists('%s/%s/' % (outpath, obs)):
        os.makedirs('%s/%s/' % (outpath, obs))

    fig, ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    for i in range(4):
        pl.image_plot(fig, ax[i / 2][i % 2], INS[:, 0, :, i], title=pols[i],
                      **image_plot_kwargs)
    fig.savefig('%s/%s/%s_INS_Filtered.png' % (outpath, obs, obs))
    plt.close(fig)

    for m, (event, hist) in enumerate(zip(events, hists)):
        title_tuple = (obs,
                       min(freqs[event[0], event[2]]) * 10 ** (-6),
                       max(freqs[event[0], event[2]]) * 10 ** (-6),
                       pols[event[1]])

        tag_tuple = (outpath,
                     obs, obs,
                     event[0],
                     event[2].indices(freqs.shape[1])[0],
                     event[2].indices(freqs.shape[1])[1],
                     pols[event[1]],
                     m)

        fig, ax = plt.subplots(figsize=(14, 8))
        pl.one_d_hist_plot(fig, ax, hist[2], hist[0:2],
                           labels=['Measurements', 'Fit'], xlog=False, ylog=False,
                           title='%s %.2f - %.2f %s' % (title_tuple),
                           xlabel='$\sigma$')
        fig.savefig('%s/%s/%s_INS_hist_spw%i_f%i_f%i_%s_%i.png' % tag_tuple)
        plt.close(fig)
