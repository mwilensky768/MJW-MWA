import rfipy
import numpy as np
import matplotlib.pyplot as plt
import plot_lib as pl
import argparse
from math import pi

parser = argparse.ArgumentParser()
parser.add_argument("inpath", action='store', nargs=1, help="The file you want to process")
parser.add_argument("outpath", action='store', nargs=1,
                    help="The target directory for plots and arrays, be sure to include the final /")
args = parser.parse_args()
obs = args.inpath[0][-17:-7]

bad_time_indices = [0, -1, -2, -3]
flag = False

RFI = rfipy.RFI(str(obs), args.inpath[0], args.outpath[0], bad_time_indices=bad_time_indices)
RFI.apply_flags(app=flag)

sig_arr, N_arr = RFI.MLE_calc(axis=0, flag=flag)
RFI.UV.data_array = 0.5 * RFI.UV.data_array**2 / sig_arr
MC_data = np.random.exponential(size=RFI.UV.data_array.shape)
MC_data = np.ma.masked_array(MC_data)
MC_data.mask = RFI.UV.data_array.mask

data_hist, bins_orig = np.histogram(RFI.UV.data_array[np.logical_not(RFI.UV.data_array.mask)], bins='auto')
MC, _ = np.histogram(MC_data[np.logical_not(MC_data.mask)], bins=bins_orig)

meas_mean = RFI.UV.data_array.mean(axis=(0, 3))[:, 0, 0]
N = float(np.amax(np.count_nonzero(np.logical_not(RFI.UV.data_array.mask), axis=(0, 3))))
print(N)
MC_mean = MC_data.mean(axis=(0, 3))[:, 0, 0]

meas_hist, bins = np.histogram(meas_mean[np.logical_not(meas_mean.mask)], bins='auto')
MC_hist, bins_MC_hist = np.histogram(MC_mean[np.logical_not(MC_mean.mask)], bins='auto')
w = np.diff(bins_MC_hist)
x = bins_MC_hist[:-1] + w
fit = (N / (N - 1))**(N - 1) * N * x**(N - 1) * w * np.sum(meas_hist) * np.exp((N * (1 - x) - 1))
print(w)
print(np.sum(MC_hist))
print(fit)


fig, ax = plt.subplots(figsize=(14, 8))
fig_orig, ax_orig = plt.subplots(figsize=(14, 8))
pl.one_d_hist_plot(fig, ax, bins, [meas_hist, ], labels=['Measurements', ],
                   xlog=False, xlabel='0.5 * Amp^2 (Sigma^2)')
pl.one_d_hist_plot(fig, ax, bins_MC_hist, [MC_hist, fit], labels=['MC', 'Analytic Fit'],
                   xlog=False, xlabel='0.5 * Amp^2 (Sigma^2)')
pl.one_d_hist_plot(fig_orig, ax_orig, bins_orig, [data_hist, MC], labels=['Measurements', 'MC'],
                   xlog=False, xlabel='0.5 * Amp^2 (Sigma^2)')


fig.savefig('/Users/mike_e_dubs/MWA/Test_Plots/1061312272_hist_MC_test_no_flag_exp_a0.png')
fig_orig.savefig('/Users/mike_e_dubs/MWA/Test_Plots/1061312272_hist_MC_test_no_flag_orig_exp_a0.png')
