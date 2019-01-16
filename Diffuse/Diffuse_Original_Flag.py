from SSINS import SS, Catalog_Plot
from pyuvdata import UVData
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('obs')
parser.add_argument('inpath')
parser.add_argument('outpath')
parser.add_argument('--times', type=int, nargs='*', help='The time indices to flag')
args = parser.parse_args()

UV = UVData()
UV.read(args.inpath, file_type='uvfits')
JD = np.unique(UV.time_array)[-2]
where = np.where(UV.time_array == JD)
UV.flag_array[where] = 1
auto_bl = np.where(ant_1_array == ant_2_array)
UV.flag_array[auto_bl] = 1

shape_dict = {'TV6': [1.74e8, 1.81e8],
              'TV7': [1.81e8, 1.88e8],
              'TV8': [1.88e8, 1.95e8],
              'broad6': [1.72e8, 1.83e8],
              'broad7': [1.79e8, 1.9e8],
              'broad8': [1.86e8, 1.97e8]}

sky_sub = SS(obs=args.obs, UV=UV, outpath=args.outpath, flag_choice='original')
sky_sub.INS_prepare()
Catalog_Plot.INS_plot(sky_sub.INS)
sky_sub.MF_prepare(sig_thresh=5, shape_dict=shape_dict, N_thresh=15)
sky_sub.apply_match_test(apply_N_thresh=True)
Catalog_Plot.INS_plot(sky_sub.INS)
