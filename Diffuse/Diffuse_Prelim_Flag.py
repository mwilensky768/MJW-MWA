from SSINS import util, INS, MF
from SSINS import Catalog_Plot as cp
import numpy as np
import glob
import pickle

obslist = util.make_obslist('/Users/mike_e_dubs/MWA/Obs_Lists/sidelobe_survey_obsIDs.txt')
indir = '/Users/mike_e_dubs/MWA/INS/Diffuse'
outpath = '%s_Filtered' % indir
sig_thresh = 5
shape_dict = {'TV6': [1.74e8, 1.81e8],
              'TV7': [1.81e8, 1.88e8],
              'TV8': [1.88e8, 1.95e8]}
mf_kwargs = {'sig_thresh': sig_thresh,
             'shape_dict': shape_dict}
occ_dict = {}
for obs in obslist:
    if len(glob.glob('%s/figs/%s*' % (indir, obs))):
        read_paths = util.read_paths_construct(indir, None, obs, 'INS')
        ins = INS(obs=obs, outpath=outpath, read_paths=read_paths)
        ins.data[-5:] = np.ma.masked
        ins.data[0] = np.ma.masked
        ins.data_ms = ins.mean_subtract()
        mf = MF(ins, **mf_kwargs)
        mf.apply_match_test()
        cp.MF_plot(mf)
        cp.INS_plot(ins, ms_vmin=-5, ms_vmax=5)
        occ_dict['obs'] = (np.count_nonzero(ins.data_ms.mask[:-3]) / ins.data[:-3].size)
        del ins
        del mf
pickle.dump(occ_dict, open('%s/occ_dict_only_subflags' % outpath, 'wb'))
