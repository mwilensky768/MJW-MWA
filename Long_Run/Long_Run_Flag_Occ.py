from __future__ import division

from SSINS import INS, MF, util
import numpy as np
import glob
import pickle
import os

obsfile = '/Users/mike_e_dubs/Repositories/MJW-MWA/Obs_Lists/Long_Run_8s_Autos_OBSIDS.txt'
basedir = '/Users/mike_e_dubs/MWA/INS/Long_Run/Original'
outdir = '%s_Jackknife' % basedir
if not os.path.exists(outdir):
    os.makedirs(outdir)
obslist = util.make_obslist(obsfile)

sig_list = [5, 10, 20, 40, 80]


shape_dict = {'TV6': [1.74e8, 1.81e8],
              'TV7': [1.81e8, 1.88e8],
              'TV8': [1.88e8, 1.95e8],
              'broad6': [1.72e8, 1.83e8],
              'broad7': [1.79e8, 1.9e8],
              'broad8': [1.86e8, 1.97e8]}
shapes = ['TV6', 'TV7', 'TV8', 'broad6', 'broad7', 'broad8', 'streak', 'point', 'total']
occ_dict = {sig: {shape: {} for shape in shapes} for sig in sig_list}

for obs in obslist:
    flist = glob.glob('%s/metadata/%s*' % (basedir, obs))
    if len(flist):
        read_paths = util.read_paths_construct(basedir, 'original', obs, 'INS')
        for sig_thresh in sig_list:
            ins = INS(read_paths=read_paths, obs=obs, outpath=outdir,
                      flag_choice='original')
            mf = MF(ins, sig_thresh=sig_thresh, N_thresh=15, shape_dict=shape_dict)
            mf.apply_match_test(apply_N_thresh=True)
            occ_dict[sig_thresh]['total'][obs] = np.mean(ins.data.mask[:, 0, :, 0], axis=0)
            if len(ins.match_events):
                event_frac = util.event_fraction(ins.match_events, ins.data.shape[0], shapes, 384)
                for shape in shapes[:-1]:
                    occ_dict[sig_thresh][shape][obs] = event_frac[shape]
            if obs == '1061312152' and sig_thresh is 5:
                pickle.dump(mf.slice_dict, open('%s/long_run_shape_dict.pik' % outdir, 'wb'))
            del ins
            del mf

print(occ_dict)

pickle.dump(occ_dict, open('%s/long_run_original_occ_dict.pik' % outdir, 'wb'))
