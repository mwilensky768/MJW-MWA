import rfipy as rfi
import Catalog_Funcs as cf
import glob
import numpy as np
from matplotlib.ticker import FixedLocator, AutoMinorLocator

testfile = '/Users/mike_e_dubs/MWA/Data/uvfits/1061313008.uvfits'
plot_outdir = '/Users/mike_e_dubs/MWA/Test_Plots/'
flag_slices = ['All', 'Unflagged']
write = {'Unflagged': True, 'All': True}
writepath = plot_outdir
band = {'All': [2e3, 1e5], 'Unflagged': [2e3, 1e5]}
fit_type = {'All': 'rayleigh', 'Unflagged': 'rayleigh'}

RFI = rfi.RFI('1061313008', testfile, auto_remove=True, bad_time_indices=[0, -1])
xticks = [RFI.UV.Nfreqs * k / 6 for k in range(6)]
xticks.append(RFI.UV.Nfreqs - 1)
xminors = AutoMinorLocator(4)

cf.waterfall_catalog(RFI, plot_outdir, band=band, fit_type=fit_type, bin_window=[0, 2e3],
                     xticks=xticks, xminors=xminors, write=write, writepath=writepath)
cf.INS_catalog(RFI, plot_outdir, flag_slices=flag_slices, xticks=xticks,
               xminors=xminors, yminors='auto', write=True, writepath=plot_outdir)
