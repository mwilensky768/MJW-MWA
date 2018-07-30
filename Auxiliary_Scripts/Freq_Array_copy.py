import glob
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('target', action='store', help='The target directory')
args = parser.parse_args()

freq_arr_path = '/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/MWA_Highband_Freq_Array.npy'
freq_arr = np.zeros([1, 384])
freq_arr[0, :] = np.load(freq_arr_path)
paths = glob.glob('%s/metadata/*' % args.target)
L = len('%s/metadata/' % args.target)
paths = np.unique([path[:L + 10] for path in paths])
for path in paths:
    np.save('%s_freq_array.npy' % path, freq_arr)
