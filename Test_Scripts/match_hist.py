import rfiutil
import numpy as np
import matplotlib.pyplot as plt
import plot_lib as pl
import os
from matplotlib.ticker import AutoMinorLocator
import scipy.stats
from scipy.special import erfc, erfcinv
import glob

indir = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Golden_Set_8s_Autos/Vis_Avg/Averages'
arr_list = glob.glob('%s/*.npy' % (indir))
arr_tag = '_Vis_Avg_Amp_All.npy'
freq_arr_path = '/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/MWA_Highband_Freq_Array.npy'
freqs = np.zeros([1, 384])
freqs[0, :] = np.load(freq_arr_path)
test = 'chisq_test'
test_kwargs = {'chisq_test': {'weight': 'var'},
               'ks_test': {}}
N = 51 * 384 * 4
sig_thresh = np.sqrt(2) * erfcinv(1. / N)
outpath = '/Users/mike_e_dubs/MWA/Test_Plots/match_filter_hist_converge_4sig/all_obs_errorbar_var_%s_%s_shape_check_tune_ind_pol_streak' % (test, sig_thresh)
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
alpha = erfc(sig_thresh / np.sqrt(2))
shape_test = True

for arr in arr_list:
    obs = arr[len(indir) + 1: len(indir) + 11]
    if not os.path.exists('%s/%s/' % (outpath, obs)):
        os.makedirs('%s/%s/' % (outpath, obs))
    INS = np.ma.masked_array(np.load(arr))
    Nbls = 8001 * np.ones(INS.shape)
    MS = (INS / INS.mean(axis=0) - 1) * np.sqrt(Nbls / C)

    INS, MS, events, hists = \
        rfiutil.match_filter(INS, MS, Nbls, outpath, freqs, shape_dict=shape_dict, samp_thresh=50)

    MS_slc = MS.mean(axis=2) * np.sqrt(np.count_nonzero(np.logical_not(MS.mask), axis=2))
    print(MS_slc.max())
    fig, ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    fig.suptitle('%s Match Filtered, MS Streaks %s' % (obs, test))
    for i in range(4):
        n, bins = np.histogram(MS_slc[:, 0, i][np.logical_not(MS_slc.mask[:, 0, i])], bins=np.linspace(-sig_thresh, sig_thresh, num=9))
        x = bins[:-1] + 0.5 * np.diff(bins)
        P = scipy.stats.norm.cdf(bins[1:]) - scipy.stats.norm.cdf(bins[:-1])
        exp = P * np.sum(n)
        var = exp * (1 - P)
        pl.error_plot(fig, ax[i / 2][i % 2], x, n, None, None, label='Measurements', drawstyle='steps-mid')
        pl.error_plot(fig, ax[i / 2][i % 2], x, exp, None, np.sqrt(var), label='Fit', drawstyle='steps-mid',
                      title=pols[i], xlabel='$\sigma$', ylabel='Counts')
    fig.savefig('%s/%s/%s_MS_streak_compare.png' % (outpath, obs, obs))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 8))
    n, bins = np.histogram(MS_slc[:, 0][np.logical_not(MS_slc.mask[:, 0])],
                           bins=np.linspace(-sig_thresh, sig_thresh, num=2 * int(np.ceil(2 * sig_thresh + 1))))
    x = bins[:-1] + 0.5 * np.diff(bins)
    P = scipy.stats.norm.cdf(bins[1:]) - scipy.stats.norm.cdf(bins[:-1])
    exp = P * np.sum(n)
    var = exp * (1 - P)
    pl.error_plot(fig, ax, x, n, None, None, label='Measurements', drawstyle='steps-mid')
    pl.error_plot(fig, ax, x, exp, None, np.sqrt(var), label='Fit', drawstyle='steps-mid',
                  title='%s Match Filtered, MS Streaks %s' % (obs, test), xlabel='$\sigma$', ylabel='Counts')
    fig.savefig('%s/%s/%s_MS_streak_allpol.png' % (outpath, obs, obs))
    plt.close(fig)

    """event_shapes = np.unique(events[:, :-1])
    event_shapes = np.delete(event_shapes, [0])
    if test:
        for slc in event_shapes:
            p_min = 1
            event = [0, slc]
            fig, ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
            fig.suptitle('%s ind pol event hist %s' % (obs, slc))
            for m in range(4):
                test_args = {'ks_test': (MS[:, :, :, m], event),
                             'chisq_test': (MS[:, :, :, m], sig_thresh, event)}
                stat, p, counts, exp, var, bins = getattr(rfiutil, test)(*test_args[test])
                x = bins[:-1] + 0.5 * np.diff(bins)
                pl.error_plot(fig, ax[m / 2][m % 2], x, counts, None, None, label='Measurements', drawstyle='steps-mid')
                pl.error_plot(fig, ax[m / 2][m % 2], x, exp, None, np.sqrt(var), label='Fit', drawstyle='steps-mid',
                              title=pols[m], xlabel='$\sigma$', ylabel='Counts')
                if p < alpha:
                    p_min = p
            if p_min < 1:
                INS[:, event[0], event[1]] = np.ma.masked
            fig.savefig('%s/%s/%s_%s.png' % (outpath, obs, obs, slc))
            plt.close(fig)"""

    fig, ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    fig.suptitle('%s Match Filtered, %s' % (obs, test))
    for i in range(4):
        pl.image_plot(fig, ax[i / 2][i % 2], INS[:, 0, :, i], title=pols[i],
                      **image_plot_kwargs)
    fig.savefig('%s/%s/%s_INS_Filtered_%s.png' % (outpath, obs, obs, test))
    plt.close(fig)

    """for m, (event, hist) in enumerate(zip(events, hists)):
        N = np.sum(hist[0])
        p = scipy.stats.norm.cdf(hist[1][1:]) - scipy.stats.norm.cdf(hist[1][:-1])
        fit = N * p
        error = np.sqrt(N * p * (1 - p))
        x = hist[1][:-1] + 0.5 * np.diff(hist[1])
        title_tuple = (obs,
                       min(freqs[event[0], event[1]]) * 10 ** (-6),
                       max(freqs[event[0], event[1]]) * 10 ** (-6))

        tag_tuple = (outpath,
                     obs, obs,
                     event[0],
                     event[1].indices(freqs.shape[1])[0],
                     event[1].indices(freqs.shape[1])[1],
                     m)

        fig, ax = plt.subplots(figsize=(14, 8))
        pl.error_plot(fig, ax, x, fit, None, error, label='Fit', drawstyle='steps-mid')
        pl.one_d_hist_plot(fig, ax, hist[1], [hist[0], ],
                           labels=['Measurements', ], xlog=False, ylog=False,
                           title='%s %.2f - %.2f' % (title_tuple),
                           xlabel='$\sigma$')
        fig.savefig('%s/%s/%s_INS_hist_spw%i_f%i_f%i_%i.png' % tag_tuple)
        plt.close(fig)"""
