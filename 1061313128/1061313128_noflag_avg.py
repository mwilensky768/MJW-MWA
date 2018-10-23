from SSINS import SS
from SSINS import Catalog_Plot as cp

inpath = '/Users/mike_e_dubs/MWA/Data/1061313128_noflag.uvfits'
outpath = '/Users/mike_e_dubs/MWA/1061313128_noflag'
obs = '1061313128'

ss = SS(inpath=inpath, outpath=outpath, obs=obs, bad_time_indices=[0, -1, -2, -3, -4],
        read_kwargs={'ant_str': 'cross', 'file_type': 'uvfits'})
ss.MF_prepare(sig_thresh=5, shape_dict={'TV6': [1.74e8, 1.81e8],
                                        'TV7': [1.81e8, 1.88e8],
                                        'TV8': [1.88e8, 1.95e8],
                                        'TV7_Broad': [1.79e8, 1.9e8]})
cp.INS_plot(ss.INS, ms_vmin=-5, ms_vmax=5)
ss.MF.apply_match_test()
cp.MF_plot(ss.MF, ms_vmin=-5, ms_vmax=5)
