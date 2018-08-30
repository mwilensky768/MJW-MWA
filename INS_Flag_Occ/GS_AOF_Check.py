from pyuvdata import UVData
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('obs')
parser.add_argument('infile')
parser.add_argument('outpath')
args = parser.parse_args()

UV = UVData()
UV.read(args.infile, type='uvfits', ant_str='cross')
UV.select(times=np.unique(time_array)[[0, -1, -2, -3]])
edges = [0 + 16 * i for i in range(24)] + [15 + 16 * k for k in range(24)]
UV.flag_array[:, :, edges] = 0
num = UV.flag_array.sum()
den = np.prod(UV.flag_array.shape)

np.save('%s/%s_num.npy' % (args.outpath, args.obs))
np.save('%s/%s_den.npy' % (args.outpath, args.obs))
