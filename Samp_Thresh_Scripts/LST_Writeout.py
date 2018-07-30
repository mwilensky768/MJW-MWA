from pyuvdata import UVData
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('infile')
parser.add_argument('outdir')
args = parser.parse_args

UV = UVData()
UV.read(infile, file_type='uvfits', read_data=False)
lst_arr = np.unique(UV.lst_array)[1:-3]
times_arr = np.unique(UV.times_arr)[1:-3]
obs = infile[8:18]
np.save('%s/%s_lst_arr.npy' % (outdir, obs), lst_arr)
np.save('%s/%s_times_arr.npy' % (outdir, obs), times_arr)
