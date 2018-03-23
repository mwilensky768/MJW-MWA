from matplotlib import use
use('Agg')
import rfipy as rfi
import Catalog_Funcs as cf
import glob
import numpy as np


testfile = '/Users/mike_e_dubs/MWA/Data/smaller_uvfits/f1061313008.uvfits'
outpath = '/Users/mike_e_dubs/MWA/Test_Plots/1061313128_pickle_test/'
flag_slices = ['All', 'Unflagged']
band = {'All': [2e3, 1e5], 'Unflagged': [2e3, 1e5]}
fit_type = {'All': 'rayleigh', 'Unflagged': 'rayleigh'}

RFI = rfi.RFI('1061313128', testfile, auto_remove=True, bad_time_indices=[0, -1])

cf.waterfall_catalog(RFI, outpath, band=band, fit_type=fit_type, bin_window=[0, 2e3])
assert(False)
cf.INS_catalog(RFI, outpath)
writepath = '%s%s/all_spw/arrs/' % (outpath, 'All')
base = '%s%s_%s_%s' % (writepath, RFI.obs, 'All', 'Amp')
INS = np.ma.load('%s_INS.npym' % (base))
cf.bl_scatter_catalog(RFI, INS.mask)
