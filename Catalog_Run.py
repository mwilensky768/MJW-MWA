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
catalog_types = ['INS', ]

"""Object Keywords"""

# The beginning and end of the obs are almost always problematic
# The autocorrelations are removed since they are much brighter than the cross-correlations and mess up the statistics
bad_time_indices = [0, -1]
auto_remove = True

"""Misc. Keywords"""

# Could add flagged to flag_slices list if you want to just look at flagged data (data identified as contaminated by COTTER)
flag_slices = ['All', 'Unflagged']
bins = 'auto'
band = {'Unflagged': 'fit', 'All': [3e+03, 1e+05], 'Flagged': [3e3, 1e6]}
fit_type = {'Unflagged': 'rayleigh', 'All': False}
bin_window = [0, 2e+03]

"""Waterfall Keywords"""

fraction = True

"""INS Keywords"""

# mask is an experimental feature which attempts to flag the noise spectra based on some statistical reasoning.
# Results may vary. It is nowhere near completion.
amp_avg = 'Amp'
invalid_mask = False
mask = True


RFI = rfi.RFI(str(obs), args.inpath[0], auto_remove=auto_remove,
              bad_time_indices=bad_time_indices)

if 'waterfall' in catalog_types:
    cf.waterfall_catalog(RFI, args.outpath[0], bins=bins, band=band,
                         flag_slices=flag_slices, fit_type=fit_type,
                         bin_window=bin_window, fraction=fraction)
if 'INS' in catalog_types:
    cf.INS_catalog(RFI, args.outpath[0], flag_slices=flag_slices, amp_avg=amp_avg,
                   invalid_mask=invalid_mask, mask=mask)
