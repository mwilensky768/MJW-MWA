import rfipy as rfi
import Catalog_Funcs as cf
import Catalog_Class as cc
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

RFI_kwargs = {}
catalog_kwargs = {}
catalog_types = ['INS', ]
data_kwargs = {'pow': 1,
               'typ': 'var',
               'match_filter': False,
               'sig_thresh': 4.5,
               'shape_dict': {'TV%i' % (i): np.load('./Useful_Information/TV%i_freqs.npy' % (i)) + np.array([-2, 2]) for i in [6, 7, 8]}}
catalog_kwargs['bl_grid'] = {'bl_grid_kwargs': {'MLE_kwargs': {'axis': 0,
                                                               'flag_kwargs': {'choice': 'INS'}},
                                                'INS_kwargs': {'match_filter': True,
                                                               'match_filter_kwargs': {'shape_dict': {'TV%i' % (7): np.load('./Useful_Information/TV%i_freqs.npy' % (7)) + np.array([-2, 2])}}}}}


RFI = rfi.RFI(str(obs), args.inpath[0], args.outpath[0], **RFI_kwargs)

for catalog_type in catalog_types:
    cc.Catalog_Generate(RFI, catalog_type, data_kwargs=data_kwargs)
