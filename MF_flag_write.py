from SSINS import SS
from SSINS import Catalog_Plot as cp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('obs')
parser.add_argument('inpath')
parser.add_argument('outpath')
args = parser.parse_args()

ss = SS(obs=args.obs, outpath=args.outpath, inpath=args.inpath, flag_choice='original',
        read_kwargs={'ant_str': 'cross'})
ss.MF_prepare(sig_thresh=5, shape_dict={'TV6': [1.74e8, 1.81e8],
                                        'TV7': [1.81e8, 1.88e8],
                                        'TV8': [1.88e8, 1.95e8]})
ss.MF.apply_match_test()
cp.MF_plot(ss.MF)
ss.write('%s/%s.uvfits' % (args.outpath, args.obs), 'uvfits', inpath=args.inpath,
         nsample_default=8, read_kwargs={'ant_str': 'cross'})
