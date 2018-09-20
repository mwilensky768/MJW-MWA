from __future__ import absolute_import, division, print_function

import argparse
from SSINS import Catalog_Plot as cp
from SSINS import SS
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('obs', action='store', help='How the observation will be referred to')
parser.add_argument('inpath', action='store', help='The path to the data file, and the file_type')
parser.add_argument('outpath', action='store', help='The base directory for saving all outputs')
parser.add_argument('--time_range', nargs=2, help='The time range NOT to read in')
args = parser.parse_args()

# Here is a dictionary for the RFI class keywords

data_kwargs = {'read_kwargs': {'file_type': 'uvfits', 'ant_str': 'cross'},
               'obs': args.obs,
               'inpath': args.inpath,
               'outpath': args.outpath}
if args.time_range is not None:
    data_kwargs['bad_time_indices'] = np.arange(min(args.time_range),
                                                max(args.time_range))

# The type of catalog you would like made - options are 'INS', 'VDH', 'MF', and 'ES'
catalog_types = ['INS', ]


catalog_data_kwargs = {'INS': {},
                       'VDH': {'fit_hist': True},
                       'MF': {'sig_thresh': 5,
                              'shape_dict': {'TV4': (1.74e8, 1.82e8),
                                             'TV5': (1.82e8, 1.9e8),
                                             'TV6': (1.9e8, 1.98e8),
                                             'TV7': (1.98e8, 2.06e8)},
                              'tests': ['match']},
                       'ES': {}}

catalog_plot_kwargs = {'INS': {},
                       'VDH': {},
                       'MF': {},
                       'ES': {}}

sky_sub = SS(**data_kwargs)


"""
Do not edit things beneath this line!
"""


for cat in catalog_types:
    getattr(sky_sub, '%s_prepare' % (cat))(**catalog_data_kwargs[cat])
    getattr(cp, '%s_plot' % (cat))(getattr(sky_sub, cat), **catalog_plot_kwargs[cat])
sky_sub.save_data()
sky_sub.save_meta()
