import rfipy as rfi
import Catalog_Funcs as cf
import argparse
import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("inpath", action='store', nargs=1, help="The file you want to process")
parser.add_argument("outpath", action='store', nargs=1,
                    help="The target directory for plots and arrays, be sure to include the final /")
args = parser.parse_args()
obs = args.inpath[0][-17:-7]

"""Input/Output keywords"""

# You can add 'waterfall' to this list to compute another type of catalog
catalog_types = ['bl_scat', ]

"""Object Keywords"""

# The beginning and end of the obs are almost always problematic
# The autocorrelations are removed since they are much brighter than the cross-correlations and mess up the statistics
bad_time_indices = [0, -1, -2, -3]
auto_remove = True

"""Misc. Keywords"""

# Could add flagged to flag_slices list if you want to just look at flagged data (data identified as contaminated by COTTER)
bins = None
amp_range = {True: 'fit', False: [2e+03, 1e+05]}
bin_window = [0, 2e+03]

"""Waterfall Keywords"""

fraction = True

"""INS Keywords"""

# mask is an experimental feature which attempts to flag the noise spectra based on some statistical reasoning.
# Results may vary. It is nowhere near completion.
invalid_mask = False
mask = True

"""Match Filter Keywords"""
shape_dict = {'TV%i' % (k): np.load('/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/TV%i_Freqs.npy' % (k)) for k in [6, 7, 8]}


RFI = rfi.RFI(str(obs), args.inpath[0], args.outpath[0], bad_time_indices=bad_time_indices)

if 'waterfall' in catalog_types:
    cf.waterfall_catalog(RFI, bins=bins, amp_range=amp_range,
                         bin_window=bin_window, fraction=fraction)
if 'INS' in catalog_types:
    cf.INS_catalog(RFI, invalid_mask=invalid_mask, mask=mask)
if 'rms' in catalog_types:
    RFI.rms_calc(flag=False)
    RFI.rms_calc()
if 'bl_scat' in catalog_types:
    cf.bl_scatter_catalog(RFI, shape_dict=shape_dict)
