from SSINS import INS, util, MF
from SSINS import Catalog_Plot as cp
import numpy as np

inpath = '/Users/mike_e_dubs/MWA/1061313128_noavg_noflag'
outpath = '%s_filtered' % inpath
obs = '1061313128'
read_paths = util.read_paths_construct(inpath, None, obs, 'INS')
print(read_paths)
MF_kwargs = {'shape_dict': {'TV6': [1.74e8, 1.81e8],
                            'TV7': [1.81e8, 1.88e8],
                            'TV8': [1.88e8, 1.95e8],
                            'TV7_broad': [1.79e8, 1.9e8]}}

ins = INS(read_paths=read_paths, obs=obs, outpath=outpath)
ins.data[[0, 1, 2, 3, -1, -2, -3, -4, -5, -6], :, :, :] = np.ma.masked
ins.data_ms = ins.mean_subtract()
cp.INS_plot(ins, ms_vmin=-5, ms_vmax=5)
mf = MF(ins, **MF_kwargs)
print(mf.sig_thresh)
mf.apply_match_test()
cp.MF_plot(mf)
