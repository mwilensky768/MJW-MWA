from pyuvdata import UVData
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('obsid')
parser.add_argument('inpath')
parser.add_argument('outpath')
parser.add_argument('TV_ind', nargs=2, type=int)
parser.add_argument('cal_ind', nargs=2, type=int)
parser.add_argument('channels', nargs='*', type=int)
args = parser.parse_args()

UV = UVData()
UV.read(args.inpath, file_type='uvfits')
freq_chans = {6: range(87, 174),
              7: range(174, 262),
              8: range(262, 349)}

for channel in args.channels:
    UV_TV = UV.select(times=np.unique(UV.time_array)[min(args.TV_ind):max(args.TV_ind)],
                      inplace=False)
    ind = np.where(UV_TV.nsample_array == 0)
    for m in range(len(ind[0])):
        UV_TV.nsample_array[ind[0][m], ind[1][m], ind[2][m], ind[3][m]] = \
            UV_TV.nsample_array[0, ind[1][m], ind[2][m] % 16, ind[3][m]]
    UV_TV.flag_array[:, :, freq_chans[channel]] = 0
    UV_TV.flag_array[:, :, :min(freq_chans[channel])] = 1
    UV_TV.flag_array[:, :, max(freq_chans[channel]):] = 1

    UV_cal = UV.select(times=np.unique(UV.time_array)[min(args.cal_ind):max(args.cal_ind)],
                       inplace=False)
    UV_cal.flag_array[:, :, freq_chans[channel]] = 0
    UV_cal.flag_array[:, :, :min(freq_chans[channel])] = 1
    UV_cal.flag_array[:, :, max(freq_chans[channel]):] = 1

    ind = np.where(UV_cal.nsample_array == 0)
    for m in range(len(ind[0])):
        UV_cal.nsample_array[ind[0][m], ind[1][m], ind[2][m], ind[3][m]] = \
            UV_cal.nsample_array[0, ind[1][m], ind[2][m] % 16, ind[3][m]]

    UV_TV.write_uvfits('%s/%s_TV%i_t%i_t%i.uvfits' %
                       (args.outpath, args.obsid, channel, min(args.TV_ind), max(args.TV_ind)))
    UV_cal.write_uvfits('%s/%s_cal%i_t%i_t%i.uvfits' %
                        (args.outpath, args.obsid, channel, min(args.cal_ind), max(args.cal_ind)))
