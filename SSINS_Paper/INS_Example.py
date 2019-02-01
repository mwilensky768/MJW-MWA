from SSINS import SS, plot_lib, util
from pyuvdata import UVData
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

indir = '/Volumes/Faramir/uvfits'
obslist = ['1061312640', '1066742016', '1061313128', '1067259120']
bins = np.linspace(-14, 14, num=113)


for obs in obslist:
    UV = UVData()
    UV.read('%s/%s.uvfits' % (indir, obs), file_type='uvfits', polarizations=-5)
    UV.select(times=np.unique(UV.time_array)[1:-3], ant_str='cross')
    ss = SS(UV=UV, outpath='/Users/mikewilensky/SSINS_Paper', obs=obs, flag_choice='original')
    ss.INS_prepare()
    fig, ax = plt.subplots(figsize=(16, 9))
    plot_lib.image_plot(fig, ax, ss.INS.data[:, 0, :, 0], aspect='auto',
                        freq_array=UV.freq_array[0], ylabel='Time (2 s)',
                        xlabel='Frequency (Mhz)', cbar_label='Amplitude (UNCALIB)')
    fig_ms, ax_ms = plt.subplots(figsize=(16, 9))
    plot_lib.image_plot(fig_ms, ax_ms, ss.INS.data_ms[:, 0, :, 0], aspect='auto',
                        freq_array=UV.freq_array[0], ylabel='Time (2 s)',
                        xlabel='Frequency (Mhz)', cmap=cm.coolwarm,
                        cbar_label='Deviation ($\hat{\sigma}$)', mask_color='black')
    fig.savefig('%s/%s_INS_original_data.pdf' % (ss.outpath, obs))
    fig_ms.savefig('%s/%s_INS_original_data_ms.pdf' % (ss.outpath, obs))

    fig_hist, ax_hist = plt.subplots(figsize=(16, 9))
    counts, bins = np.histogram(ss.INS.data_ms[:, 0, :, 0], bins=bins)
    exp_counts, exp_var = util.hist_fit(counts, bins)
    counts = np.append(counts, 0)
    exp_counts = np.append(exp_counts, 0)
    exp_var = np.append(exp_var, 0)

    plot_lib.error_plot(fig_hist, ax_hist, bins, counts,
                        xlabel='Deviation ($\hat{\sigma}$)', ylabel='Counts',
                        yscale='log', drawstyle='steps-post',
                        ylim=[0.5, 10**(int(np.log10(np.amax(exp_counts))) + 1)],
                        label='Measurements', legend=True)
    plot_lib.error_plot(fig_hist, ax_hist, bins, exp_counts, yerr=np.sqrt(exp_var),
                        xlabel='Deviation ($\hat{\sigma}$)', ylabel='Counts',
                        yscale='log', drawstyle='steps-post', ylim=[0.5, 10**(int(np.log10(np.amax(exp_counts))) + 1)],
                        label='Model', legend=True)

    fig_hist.savefig('%s/%s_INS_original_data_ms_hist.pdf' % (ss.outpath, obs))
    ss.MF_prepare(sig_thresh=5, streak=False)
    ss.MF.apply_match_test()

    fig_ms_of, ax_ms_of = plt.subplots(figsize=(16, 9))
    plot_lib.image_plot(fig_ms_of, ax_ms_of, ss.INS.data_ms[:, 0, :, 0], aspect='auto',
                        freq_array=UV.freq_array[0], ylabel='Time (2 s)',
                        xlabel='Frequency (Mhz)', cmap=cm.coolwarm,
                        cbar_label='Deviation ($\hat{\sigma}$)', mask_color='black')
    fig_ms_of.savefig('%s/%s_INS_original_data_ms_of.pdf' % (ss.outpath, obs))

    ss.INS.data.mask[:] = 0
    ss.INS.data_ms = ss.INS.mean_subtract()

    del ss.MF
    ss.MF_prepare(sig_thresh=5, shape_dict={'TV6': [1.74e8, 1.81e8],
                                            'TV7': [1.81e8, 1.88e8],
                                            'TV8': [1.88e8, 1.95e8],
                                            'broad6': [1.72e8, 1.83e8],
                                            'broad7': [1.79e8, 1.9e8],
                                            'broad8': [1.86e8, 1.97e8]})
    ss.MF.apply_match_test()
    fig_ms_mf, ax_ms_mf = plt.subplots(figsize=(16, 9))
    plot_lib.image_plot(fig_ms_mf, ax_ms_mf, ss.INS.data_ms[:, 0, :, 0], aspect='auto',
                        freq_array=UV.freq_array[0], ylabel='Time (2 s)',
                        xlabel='Frequency (Mhz)', cmap=cm.coolwarm,
                        cbar_label='Deviation ($\hat{\sigma}$)', mask_color='black')
    fig_ms_mf.savefig('%s/%s_INS_original_data_ms_mf.pdf' % (ss.outpath, obs))
    plt.close(fig_ms_mf)
    plt.close(fig_ms_of)
    plt.close(fig)
    plt.close(fig_ms)
    plt.close(fig_hist)
    del ss
