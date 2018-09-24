from SSINS import INS
from SSINS import util
from SSINS import MF
from SSINS import Catalog_Plot as cp
import numpy as np

# 'TV7_ext': [1.845e8 - 5.15e6, 1.845e8 + 5.15e6]
basedir = '/Users/mike_e_dubs/MWA/INS/Long_Run/All'
obs = '1066743352'
outpath = '/Users/mike_e_dubs/MWA/INS/Datacon'
flag_choice = 'None'
read_paths = util.read_paths_construct(basedir, flag_choice, obs, 'INS')
shape_dict = {'TV6': [1.74e8, 1.81e8],
              'TV7': [1.81e8, 1.88e8],
              'TV8': [1.88e8, 1.95e8]}
order = 0
ins = INS(obs=obs, read_paths=read_paths, outpath=outpath, order=order)
mf = MF(ins, shape_dict=shape_dict, sig_thresh=5)
mf.apply_match_test(order=order)
ins.data.mask = False
ins.data_ms = ins.mean_subtract(order=order)
ins.outpath = '%s_0' % outpath
cp.INS_plot(ins, ms_vmax=4, ms_vmin=-4)
for i, event in enumerate(ins.match_events):
    ins.outpath = '%s_%i' % (outpath, i + 1)
    ins.data[tuple(event)] = np.ma.masked
    ins.data_ms = ins.mean_subtract(order=order)
    cp.INS_plot(ins, ms_vmax=4, ms_vmin=-4)
