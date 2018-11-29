from __future__ import division

from pyuvdata import UVData
import argparse
from SSINS import SS
from SSINS import Catalog_Plot as cp
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('obs')
parser.add_argument('basedir')
parser.add_argument('outdir')
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

UV1 = UVData()
UV2 = UVData()

UV1.read('%s/%s.xx.HH.uv' % (args.basedir, args.obs), file_type='miriad')
UV2.read('%s/%s.yy.HH.uv' % (args.basedir, args.obs), file_type='miriad')

UV = UV1 + UV2

ss = SS(UV=UV, obs=args.obs, outpath=args.outdir)

auto_bls = UV.ant_1_array[:UV.Nbls] == UV.ant_2_array[:UV.Nbls]
custom = np.zeros_like(ss.UV.flag_array)
custom[:, auto_bls] = 1
ss.apply_flags(choice='custom', custom=custom)

shape_dict = {'TV4': [1.74e8, 1.82e8],
              'TV5': [1.82e8, 1.9e8],
              'TV6': [1.9e8, 1.98e8],
              'dig1': [1.125e8, 1.15625e8],
              'dig2': [1.375e8, 1.40625e8],
              'dig3': [1.625e8, 1.65625e8],
              'dig4': [1.875e8, 1.90625e8]}

ss.INS_prepare()
ss.INS.save()

cp.INS_plot(ss.INS, ms_vmin=-5, ms_vmax=5, vmin=0, vmax=0.03, aspect=ss.UV.Nfreqs / ss.UV.Ntimes)

ss.INS.data[:, 0, :82] = np.ma.masked
ss.INS.data[:, 0, -21:] = np.ma.masked
ss.INS.data_ms = ss.INS.mean_subtract(order=1)
ss.MF_prepare(sig_thresh=5, N_thresh=15, shape_dict=shape_dict)
ss.MF.apply_match_test(apply_N_thresh=True, order=1)
ss.INS.data_ms = ss.INS.mean_subtract(order=0)
ss.MF.save()

cp.MF_plot(ss.MF, ms_vmin=-5, ms_vmax=5, vmin=0, vmax=0.03, aspect=ss.UV.Nfreqs / ss.UV.Ntimes)

ss.apply_flags(choice='INS', INS=ss.INS)
ss.UV.data_array.mask[:, auto_bls] = True

ss.write('%s/%s.HH.uvfits' % (args.outdir, args.obs), 'uvfits', UV=UV1 + UV2)
