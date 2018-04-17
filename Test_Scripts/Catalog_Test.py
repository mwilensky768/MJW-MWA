from matplotlib import use
use('Agg')
import rfipy as rfi
import Catalog_Funcs as cf
import glob
import numpy as np

testfile = '/Users/mike_e_dubs/MWA/Data/uvfits/1061313128.uvfits'
outpath = '/Users/mike_e_dubs/MWA/Test_Plots/1061313128_event_test/'
flag_slices = ['All', 'Unflagged']
band = {'All': [2e3, 1e5], 'Unflagged': [2e3, 1e5]}
fit_type = {'All': 'rayleigh', 'Unflagged': 'rayleigh'}

RFI = rfi.RFI('1061313128', testfile, auto_remove=True, bad_time_indices=[0, -1, -2])

# cf.INS_catalog(RFI, outpath, flag_slices=flag_slices)
frac_diff = np.load('/Users/mike_e_dubs/MWA/Test_Plots/1061313128_pickle_test/All/all_spw/arrs/1061313128_All_Amp_INS_mask.npym')
cf.bl_scatter_catalog(RFI, outpath, frac_diff.mask)
