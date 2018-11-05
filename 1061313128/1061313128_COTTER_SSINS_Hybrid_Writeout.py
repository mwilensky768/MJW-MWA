import argparse
from SSINS import SS
import sys

parser = argparse.ArgumentParser()
parser.add_argument('obs')
parser.add_argument('inpath')
parser.add_argument('outpath')
args = parser.parse_args()

ss = SS(obs=args.obs, inpath=args.inpath, outpath=args.outpath)

auto_bls = ss.UV.ant_1_array[:Nbls] == ss.UV.ant_2_array[:Nbls]
custom = np.copy(ss.UV.flag_array)
custom[:, auto_bls] = 1
custom[-4:-1] = 1
ss.apply_flags(choice='custom', custom=custom)

ss.MF_prepare(sig_thresh=5, shape_dict={'TV6': [1.74e8, 1.81e8],
                                        'TV7': [1.81e8, 1.88e8],
                                        'TV8': [1.88e8, 1.95e8],
                                        'TV7_broad': [1.79e8, 1.9e8]})
ss.MF.apply_match_test()
ss.apply_flags(choice='INS', INS=ss.INS)
ss.write(args.inpath, 'uvfits', inpath=args.inpath, nsample_default=8)
