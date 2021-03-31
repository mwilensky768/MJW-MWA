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


def find_freq_lims(freq_low, freq_high, freq_array):

    chan_low = np.argmin(np.abs(freq_array - freq_low))
    chan_high = np.argmin(np.abs(freq_array - freq_high))

    return(chan_low, chan_high)


#DTV7_low = 1.81e8
#DTV7_high = 1.88e8
#chan_low, chan_high = find_freq_lims(DTV7_low, DTV7_high, uv.freq_array)

#DTV6_low = 1.74e8
#DTV6_high = 1.81e8
#chan_low, chan_high = find_freq_lims(DTV6_low, DTV6_high, uv.freq_array)

Double_DTV_low = 1.74e8
Double_DTV_high = 1.88e8
chan_low, chan_high = find_freq_lims(Double_DTV_low, Double_DTV_high, uv.freq_array)

uv.data_array[:, :, :chan_low] = 0
uv.data_array[:, :, chan_high:] = 0
uv.data_array = 1e-3 * uv.data_array

uv.nsample_array[uv.nsample_array == 0] = np.amax(uv.nsample_array)
uv.flag_array[:] = False

uv.write_uvfits(args.filepath_out)
