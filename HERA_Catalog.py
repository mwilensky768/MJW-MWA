import rfipy as rfi
import numpy as np
import matplotlib.pyplot as plt
import glob

outpath = '/data4/mwilensky/catalogs/golden_set/freq_time_2/'
plotlist = glob.glob(outpath + '*.png')
flag_slices = ['All','Unflagged' ]
if plotlist:
    proc_start = len(plotlist) / len(flag_slices)
    pathlist = glob.glob('/data6/HERA/data/2458042/*.uvOR')[proc_start:]
else:
    pathlist = glob.glob('/data6/HERA/data/2458042/*.uvOR')
outpath = '/data4/mwilensky/catalogs/golden_set/freq_time_1/'
catalog_type = 'waterfall'
plot_type = 'freq-time'
band = [10**(-0.5), 10**(5)]
fit = True
fit_window = [0, 10**12]
bin_window = [10**(-6), 10**(-1)]
bad_time_indices = []
auto_remove = True
good_freq_indices = range(64,960)
bins = np.logspace(-5, 3, num=1001)
temp_write = True
write = True
writepath = '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Temperatures_HERA/Hists_Midband_Autos/'
ant_pol_times = range(55)
ant_pol_freqs = [316, 317, 318, 319, 320, 321, 322, 406, 787, 788, 849, 869, 870]

print('There are ' + str(len(pathlist)) + ' files to process ')

for path in pathlist:
    start = path.find('zen.')
    end = path.find('.uvOR')
    obs = path[start:end]

    RFI = rfi.RFI(obs, path, filetype='miriad', bad_time_indices=bad_time_indices,
                  auto_remove=auto_remove, good_freq_indices=good_freq_indices)

    if catalog_type is 'waterfall':

        RFI.rfi_catalog(outpath, band=band, flag_slices=flag_slices, fit=fit,
                        fit_window=fit_window, bin_window=bin_window, plot_type=plot_type)

    elif catalog_type is 'temperature':

        RFI.one_d_hist_prepare(flag_slice=flag_slices[0], bins=bins, fit=fit,
                               bin_window=bin_window, fit_window=fit_window,
                               temp_write=temp_write, write=write,
                               writepath=writepath)

    elif plot_type is 'ant-pol':

        RFI.ant_pol_catalog(outpath, ant_pol_times, ant_pol_freqs)
