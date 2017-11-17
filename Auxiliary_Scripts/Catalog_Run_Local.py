import rfipy_restructure as rfi
import Catalog_Funcs as cf
import numpy as np
from matplotlib.ticker import AutoMinorLocator

# bad_blt_inds = np.load('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/bad_blt_inds.npy')
# bad_blt_inds = bad_blt_inds.tolist()

obs = '1061313008'
outpath = '/Users/mike_e_dubs/MWA/Misc/Restructure_Test/'

RFI = rfi.RFI(obs,
              '/Users/mike_e_dubs/MWA/Data/smaller_uvfits/s1061313008.uvfits',
              filetype='uvfits', auto_remove=True)

xticks = [RFI.UV.Nfreqs * k / 6 for k in range(6)]
xticks.append(RFI.UV.Nfreqs - 1)
xminors = AutoMinorLocator(4)

cf.vis_avg_catalog(RFI, '/Users/mike_e_dubs/MWA/Misc/Restructure_Test/', flag_slice='All',
                   band=(2.25e+03, 10**5), xticks=xticks, yminors='auto', xminors=xminors)
