import rfipy as rfi
import numpy as np
import matplotlib.pyplot as plt

# bad_blt_inds = np.load('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/bad_blt_inds.npy')
# bad_blt_inds = bad_blt_inds.tolist()

obs = 'zen.2457555.40356.xx.HH'
inpath = '/Users/mike_e_dubs/python_stuff/miriad/temp_HERA_data/' + obs + '.uvc'
outpath = '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/'
catalog_type = 'waterfall'
plot_type = 'ant-time'
band = [10**(-0.5), 10**3]
fit = True
fit_window = [0, 10**12]
bin_window = [0, 0.2]
flag_slice = 'All'
bad_time_indices = []
auto_remove = True
good_freq_indices = range(64, 960)

RFI = rfi.RFI(obs, inpath, filetype='miriad', auto_remove=True,
              bad_time_indices=bad_time_indices, good_freq_indices=good_freq_indices)

H = RFI.one_d_hist_prepare(flag_slice=flag_slice, bins='auto', fit=fit,
                           fit_window=fit_window, bin_window=bin_window)

fig, ax = plt.subplots(nrows=2, figsize=(14, 8))

RFI.one_d_hist_plot(fig, ax[0], H, 'HERA obs ' + obs + ' Rayleigh Fit Test',
                    res_ax=ax[1])

fig.savefig(outpath + obs + 'Rayleigh_Fit_Test_3.png')
