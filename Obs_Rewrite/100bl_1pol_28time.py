from __future__ import absolute_import, print_function, division

from pyuvdata import UVData
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('inpath', action='store', help='The obs you want to re-write')
parser.add_argument('outpath', action='store', help='The final written file path')
parser.add_argument('file_type', action='store', help='The type of file')
args = parser.parse_args()

UV = UVData()
times = np.load('/Users/mike_e_dubs/python_stuff/MJW-MWA/Obs_Rewrite/1061313128_half_time.npy')
bls = [(0, k) for k in range(1, 101)]
bls.remove((0, 76))
UV.read(args.inpath, file_type=args.file_type, polarizations=-5, times=times, bls=bls)
print(UV.Ntimes)
print(UV.Nbls)

getattr(UV, 'write_%s' % (args.file_type))(args.outpath)

UV2 = UVData()
UV2.read(args.outpath, file_type=args.file_type)
