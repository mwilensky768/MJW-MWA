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

RFI_kwargs = {}
catalog_types = ['bl_grid', ]
catalog_kw_list = [{'bl_grid_kwargs': {}, 'INS_kwargs': {},
                    'match_filter_kwargs': {'shape_dict': {'TV%i' % (k): np.load('./Useful_Information/TV%i_freqs.npy' % (k))
                                                           for k in range(6, 9)}}}, ]


RFI = rfi.RFI(str(obs), args.inpath[0], args.outpath[0], **RFI_kwargs)

for catalog_type, catalog_kwargs in zip(catalog_types, catalog_kw_list):
    getattr(cf, catalog_type)(RFI, **catalog_kwargs)
