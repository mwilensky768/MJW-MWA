from SSINS import SS, util
from SSINS import Catalog_Plot as cp
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('obs')
parser.add_argument('inpath')
parser.add_argument('outpath')
args = parser.parse_args()

ss = SS(inpath=args.inpath, outpath=args.outpath, obs=args.obs, flag_choice='original',
        bad_time_indices=[0, -1, -2, -3, -4, -5], read_kwargs={'ant_str': 'cross'})
ss.INS_prepare()
ss.INS.save()
cp.INS_plot(ss.INS)
for sig_thresh in [5, 10, 20, 40, 80]:
    ss.MF_prepare(sig_thresh=sig_thresh, N_thresh=15,
                  shape_dict={'TV6': [1.74e8, 1.81e8],
                              'TV7': [1.81e8, 1.88e8],
                              'TV8': [1.88e8, 1.95e8],
                              'broad6': [1.72e8, 1.83e8],
                              'broad7': [1.79e8, 1.9e8],
                              'broad8': [1.86e8, 1.97e8]})
    ss.MF.apply_match_test()
    cp.INS_plot(ss.INS, sig_thresh=sig_thresh)
    del ss.INS
    del ss.MF
    ss.INS_prepare()
