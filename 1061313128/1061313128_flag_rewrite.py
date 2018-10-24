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
slc = slice(-4 * UV.Nbls, -1 * UV.Nbls)
UV.nsample_array[slc][UV.nsample_array[slc] == 0] = 8
UV.flag_array[slc] = 1
UV.write_uvfits('%s/1061313128.uvfits' % args.outpath)
