import numpy as np
from pyuvdata import UVData
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('fhd_output_dir')
parser.add_argument('filepath_out')
args = parser.parse_args()

uv = UVData()

filelist = glob.glob('%s/vis_data/*' % args.fhd_output_dir) + glob.glob('%s/metadata/*' % args.fhd_output_dir)
uv.read_fhd(filelist=filelist, use_model=True)

uv.nsample_array[uv.nsample_array == 0] = np.amax(uv.nsample_array)
uv.flag_array[:] = False

uv.write_uvfits(args.filepath_out)
