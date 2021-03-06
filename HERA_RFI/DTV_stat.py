import argparse
import numpy as np
from SSINS import INS, MF
from SSINS.util import make_obslist
from pyuvdata import UVData, UVFlag
import yaml

parser = argparse.ArgumentParser()
parser.add_argument('--ins_file', '-i', help='path to saved out SSINS')
parser.add_argument('--yml_file', '-y', help='Path to match_events_file')
parser.add_argument('--raw_file', '-r', help='Path to raw data file')
parser.add_argument('--outfile', '-o', help='The file to write the stat to')
args = parser.parse_args()

ins = INS(args.ins_file, match_events_file=args.yml_file)

uv = UVData()
uv.read(args.raw_file)
auto_uv = uv.select(ant_str='autos', inplace=False)
uv.select(ant_str='crosses')

shapes = ['TV4', 'TV5', 'TV6']
shape_dict = {'TV4': [1.74e8, 1.82e8],
              'TV5': [1.82e8, 1.9e8],
              'TV6': [1.9e8, 1.98e8]}

mf = MF(ins.freq_array, 5, shape_dict=shape_dict, streak=False, narrow=False)

for event in ins.match_events:
    if event[-2] in shapes:
        ins.metric_array[event[:2]] = np.ma.masked

ins_flags = ins.mask_to_flags()
for uvd in [auto_uv, uv]:
    uvf = UVFlag(uvd, mode='flag', waterfall=True)
    uvf.flag_array = ins_flags
    uvf.to_baseline(uvd)
    uvd.flag_array = uvf.flag_array

stat_dict = {'TV4': {'occ': 0, 'autopow': 0, 'crosspow': 0},
             'TV5': {'occ': 0, 'autopow': 0, 'crosspow': 0},
             'TV6': {'occ': 0, 'autopow': 0, 'crosspow': 0}}

for shape in shapes:
    stat_dict[shape]['occ'] = np.count_nonzero(np.any(ins_flags[:, mf.slice_dict[shape]], axis=(1, 2))) / float(ins_flags.shape[0])
    if stat_dict[shape]['occ'] > 0:
    	inds = (slice(None), slice(None), mf.slice_dict[shape])
    	stat_dict[shape]['autopow'] = np.mean(np.abs(auto_uv.data_array[inds][auto_uv.flag_array[inds]]))
    	stat_dict[shape]['crosspow'] = np.mean(np.abs(uv.data_array[inds][uv.flag_array[inds]]))
    else:
        stat_dict[shape]['autopow'] = 0
        stat_dict[shape]['crosspow'] = 0


print(stat_dict)
with open(args.outfile, 'w') as file:
    yaml.dump(stat_dict, file, default_flow_style=False)
