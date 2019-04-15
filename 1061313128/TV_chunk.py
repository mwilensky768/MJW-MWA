from pyuvdata import UVData
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('inpath')
parser.add_argument('outpath')
args = parser.parse_args()

chan_min = 174
chan_max = 262


for time in np.arange(11, 27):
    UV = UVData()
    UV.read(args.inpath, read_data=False)
    times = [np.unique(UV.time_array)[time]]
    UV.read(args.inpath, times=times)
    UV.nsample_array[UV.nsample_array == 0] = 1
    UV.flag_array[:] = 1
    UV.flag_array[:, :, chan_min:chan_max] = 0
    UV.write_uvfits('%s/1061313128_t%i.uvfits' % (args.outpath, time))
