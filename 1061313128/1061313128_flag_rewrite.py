from pyuvdata import UVData
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('obs')
parser.add_argument('inpath')
parser.add_argument('outpath')
args = parser.parse_args()

UV = UVData()
UV.read(args.inpath)
UV.nsample_array[UV.nsample_array == 0] = 8
MF_kwargs = {'shape_dict': {'TV6': [1.74e8, 1.81e8],
                            'TV7': [1.81e8, 1.88e8],
                            'TV8': [1.88e8, 1.95e8],
                            'TV7_broad': [1.79e8, 1.9e8]},
             'sig_thresh': 5}
UV.flag_array = UV.flag_array.reshape([UV.Ntimes, UV.Nbls, UV.Nspws, UV.Nfreqs, UV.Npols])
UV.flag_array[-4:-1] = 1
UV.flag_array = UV.flag_array.reshape([UV.Nblts, UV.Nspws, UV.Nfreqs, UV.Npols])
UV.write_uvfits('%s/1061313128.uvfits' % args.outpath)
