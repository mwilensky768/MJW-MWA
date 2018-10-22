from SSINS import SS
from SSINS import Catalog_Plot as cp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('obs')
parser.add_argument('inpath')
parser.add_argument('outpath')
args = parser.parse_args()

read_kwargs = {'ant_str': 'cross',
               'file_type': 'uvfits'}

MF_kwargs = {'sig_thresh': 5,
             'shape_dict': {'TV6': [1.74e8, 1.81e8],
                            'TV7': [1.81e8, 1.88e8],
                            'TV8': [1.88e8, 1.95e8]},
             'N_thresh': 20}

ss = SS(obs=args.obs, inpath=args.inpath, outpath=args.outpath, read_kwargs=read_kwargs,
        bad_time_indices=[0, -1, -2, -3])
ss.MF_prepare(**MF_kwargs)
cp.MF_plot(ss.MF)
