import rfipy as rfi
import numpy as np
import matplotlib.pyplot as plt
import glob

pathlist = glob.glob('/Users/mike_e_dubs/python_stuff/miriad/temp_HERA_data/*.uvc')
outpath = '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Drill_Plots/HERA/Ant-Freq/'
catalog_type = 'temperature'
plot_type = 'ant-freq'
band = [0.8, 10**3]
fit = True
fit_window = [0, 10**12]
bin_window = [0, 0.2]
flag_slices = ['All', ]
bad_time_indices = []
auto_remove = True
good_freq_indices = range(64, 960)
bins = np.logspace(-5, 3, num=1001)
temp_write = True
write = True
writepath = '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Temperatures_HERA/'

for path in pathlist:
    start = path.find('zen.')
    end = path.find('.uvc')
    obs = path[start:end]

    RFI = rfi.RFI(obs, path, filetype='miriad', bad_time_indices=bad_time_indices,
                  auto_remove=auto_remove, good_freq_indices=good_freq_indices)

    if catalog_type is 'waterfall':

        RFI.rfi_catalog(outpath, band=band, flag_slices=flag_slices, fit=fit,
                        fit_window=fit_window, bin_window=bin_window)

    elif catalog_type is 'drill':

        RFI.catalog_drill(outpath, plot_type=plot_type, band=band, fit=fit,
                          fit_window=fit_window, flag_slices=flag_slices,
                          bin_window=bin_window)

    elif catalog_type is 'temperature':

        RFI.one_d_hist_prepare(flag_slice=flag_slices[0], bins=bins, fit=fit,
                               bin_window=bin_window, fit_window=fit_window,
                               temp_write=temp_write, write=write,
                               writepath=writepath)
