import numpy as np
from pyuvdata import UVData
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('fhd_output_dir')
parser.add_argument('filepath_out')
args = parser.parse_args()

uv_fhd = UVData()
uv_rfi = UVData()

filelist = glob.glob('%s/vis_data/*' % args.fhd_output_dir) + glob.glob('%s/metadata/*' % args.fhd_output_dir)
uv_fhd.read_fhd(filelist=filelist, use_model=True)

uv_rfi = uv_fhd.copy()


def find_freq_lims(freq_low, freq_high, freq_array):

    chan_low = np.argmin(np.abs(freq_array - freq_low))
    chan_high = np.argmin(np.abs(freq_array - freq_high))

    return(chan_low, chan_high)


DTV7_low = 1.81e8
DTV7_high = 1.88e8
chan_low, chan_high = find_freq_lims(DTV7_low, DTV7_high, uv_rfi.freq_array)

uv_rfi.data_array[:, :, :chan_low] = 0
uv_rfi.data_array[:, :, chan_high:] = 0

uv_rfi.nsample_array[uv_rfi.nsample_array == 0] = np.amax(uv_rfi.nsample_array)
uv_rfi.flag_array[:] = False

uv_rfi.write_uvfits(args.filepath_out)
