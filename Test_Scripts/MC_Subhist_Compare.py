import rfipy
import plot_lib as pl
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("inpath", action='store', nargs=1, help="The file you want to process")
parser.add_argument("outpath", action='store', nargs=1,
                    help="The target directory for plots and arrays, be sure to include the final /")
args = parser.parse_args()
obs = args.inpath[0][-17:-7]

bad_time_indices = [0, -1, -2, -3]
flag = False
axis = 0

RFI = rfipy.RFI(str(obs), args.inpath[0], args.outpath[0], bad_time_indices=bad_time_indices,
                ant_str='Cross', polarizations=[-5, ])
RFI.apply_flags(app=flag)

MLE, N = RFI.MLE_calc(axis=axis, flag=flag)
# RFI.UV.data_array = RFI.UV.data_array / np.sqrt(MLE)

MC_Data = np.ma.masked_array(np.random.rayleigh(size=RFI.UV.data_array.shape,
                                                scale=np.sqrt(MLE)))
MC_Data.mask = RFI.UV.data_array.mask

obs_hist, obs_bins = np.histogram(RFI.UV.data_array[np.logical_not(RFI.UV.data_array.mask)], bins='auto')
MC_hist, MC_bins = np.histogram(MC_Data[np.logical_not(MC_Data.mask)], bins='auto')

paths = ['hists', 'figs']
sub_paths = ['total_hists', 'sub_hists']

for path in paths:
    for sub_path in sub_paths:
        total_path = '%s%s/%s/' % (args.outpath[0], path, sub_path)
        if not os.path.exists(total_path):
            os.makedirs(total_path)

np.save('%shists/total_hists/%s_obs_hist.npy' % (args.outpath[0], obs), obs_hist)
np.save('%shists/total_hists/%s_MC_hist.npy' % (args.outpath[0], obs), MC_hist)
np.save('%shists/total_hists/%s_obs_bins.npy' % (args.outpath[0], obs), obs_bins)
np.save('%shists/total_hists/%s_MC_bins.npy' % (args.outpath[0], obs), MC_bins)

fig, ax = plt.subplots(figsize=(14, 8))

pl.one_d_hist_plot(fig, ax, obs_bins, [obs_hist, ], labels=['Measurements', ],
                   xlog=True, xlabel=r'$|V|$ (UNCALIB)',
                   ylabel='Counts', title='Visibility Power MC Measurement Comparison')
pl.one_d_hist_plot(fig, ax, MC_bins, [MC_hist, ], labels=['MC', ],
                   xlog=True, xlabel=r'$|V|$ (UNCALIB)',
                   ylabel='Counts', title='Visibility Power MC Measurement Comparison')

fig.savefig('%sfigs/total_hists/%s_total_hist.png' % (args.outpath[0], obs))
plt.close(fig)

axes = (0, 3, (0, 3))
titles = ('Time', 'Frequency', 'Time-Frequency')
tags = ('t', 'f', 'tf')
slices = ((slice(None), 0, slice(None), 0),
          (slice(None), slice(None), 0, 0),
          (slice(None), 0, 0))


for axis, title, tag, slice in zip(axes, titles, tags, slices):

    fig_var, ax_var = plt.subplots(figsize=(14, 8))
    fig_mean, ax_mean = plt.subplots(figsize=(14, 8))
    fig_hist, ax_hist = plt.subplots(figsize=(14, 8))

    var1 = RFI.UV.data_array.var(axis=axis)
    MC_var1 = MC_Data.var(axis=axis)

    mean1 = RFI.UV.data_array.mean(axis=axis)
    MC_mean1 = MC_Data.mean(axis=axis)

    meas_hist, meas_bins = np.histogram(mean1, bins='auto')
    MC_hist, MC_bins = np.histogram(MC_mean1, bins=meas_bins)

    if tag is 'tf':

        pl.line_plot(fig_var, ax_var, [var1[slice], MC_var1[slice]],
                     title='Baseline Variance Comparison %s Averaged' % (title),
                     xlabel='Baseline #',
                     ylabel='Variance', labels=['Measurements', 'MC'],
                     legend=True)

        pl.line_plot(fig_mean, ax_mean, [mean1[slice], MC_mean1[slice]],
                     title='Baseline Mean Comparison %s Averaged' % (title),
                     xlabel='Baseline #', ylabel='Mean',
                     labels=['Measurements', 'MC'], legend=True)

    pl.one_d_hist_plot(fig_hist, ax_hist, meas_bins, [meas_hist, MC_hist],
                       labels=['Measurements', 'MC'], xlog=False,
                       xlabel=r'$|V|$ (UNCALIB)',
                       title='%s Averages' % (title), legend=True)

    # pl.one_d_hist_plot(fig_hist, ax_hist, MC_bins, [MC_hist, ],
                       # labels=['MC', ], xlog=False,
                       # xlabel=r'$\frac{1}{2}|V|^2$ $(\sigma^2)$',
                       # title='%s Averages' % (title), legend=True)

    fig_var.savefig('%sfigs/total_hists/%s_var_plot_%s.png' % (args.outpath[0], obs, tag))
    fig_mean.savefig('%sfigs/total_hists/%s_mean_plot_%s.png' % (args.outpath[0], obs, tag))
    fig_hist.savefig('%sfigs/sub_hists/%s_subhist_%s.png' % (args.outpath[0], obs, tag))

    plt.close(fig_var)
    plt.close(fig_mean)
    plt.close(fig_hist)
