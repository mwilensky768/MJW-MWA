import argparse
from pyuvdata import UVData
import glob
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('RFI_filepath', help='Filepath to RFI FHD dir')
parser.add_argument('GLEAM_filepath', help='Filepath to GLEAM FHD dir')
parser.add_argument('obsid', help="The obsid of the dummy obs file")
parser.add_argument('outdir', help="output directory")
parser.add_argument('name', help="The name of the file")
args = parser.parse_args()

RFI_uv = UVData()
GLEAM_uv = UVData()

RFI_filelist = glob.glob('%s/vis_data/*' % args.RFI_filepath)
RFI_filelist.append('%s/metadata/%s_settings.txt' % (args.RFI_filepath, args.obsid))
RFI_filelist.append('%s/metadata/%s_params.sav' % (args.RFI_filepath, args.obsid))

GLEAM_filelist = glob.glob('%s/vis_data/*' % args.GLEAM_filepath)
GLEAM_filelist.append('%s/metadata/%s_nsamplemax_gleam_settings.txt' % (args.GLEAM_filepath, args.obsid))
GLEAM_filelist.append('%s/metadata/%s_nsamplemax_params.sav' % (args.GLEAM_filepath, args.obsid))

RFI_uv.read_fhd(filelist=RFI_filelist, use_model=True)
GLEAM_uv.read_fhd(filelist=GLEAM_filelist, use_model=True)

low_chan = np.argmin(np.abs(RFI_uv.freq_array[0] - 1.81e8))
high_chan = np.argmin(np.abs(RFI_uv.freq_array[0] - 1.88e8))
RFI_uv.data_array[:, :, :low_chan] = 0
RFI_uv.data_array[:, :, high_chan:] = 0
RFI_uv.data_array += GLEAM_uv.data_array

RFI_uv.nsample_array[RFI_uv.nsample_array == 0] = np.amax(RFI_uv.nsample_array)

RFI_uv.write_uvfits('%s/%s_nsamplemax_RFI_plus_gleam_%s.uvfits' % (args.outdir, args.obsid, args.name),
                    spoof_nonessential=True)
