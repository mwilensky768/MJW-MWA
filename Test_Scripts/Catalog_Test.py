import rfipy as rfi
import Catalog_Funcs as cf
import glob
import numpy as np
from matplotlib import use
use('Agg')
from matplotlib.ticker import FixedLocator, AutoMinorLocator

testfile = '/Users/mike_e_dubs/MWA/Data/uvfits/1061313128.uvfits'
outpath = '/Users/mike_e_dubs/MWA/Test_Plots/1061313238_new_test/'
flag_slices = ['All', 'Unflagged']
band = {'All': [2e3, 1e5], 'Unflagged': [2e3, 1e5]}
fit_type = {'All': 'rayleigh', 'Unflagged': 'rayleigh'}

RFI = rfi.RFI('1061313128', testfile, auto_remove=True, bad_time_indices=[0, -1])

cf.waterfall_catalog(RFI, outpath, band=band, fit_type=fit_type, bin_window=[0, 2e3])
cf.INS_catalog(RFI, outpath, flag_slices=flag_slices)
