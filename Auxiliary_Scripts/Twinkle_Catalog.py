import rfipy
import plot_lib
import Catalog_Funcs as cf
from matplotlib.ticker import AutoMinorLocator
import numpy as np

obspath = '/Users/mike_e_dubs/MWA/Data/smaller_uvfits/1061313128f_180_190.uvfits'
obs = '1061313128'
outpath = '/Users/mike_e_dubs/MWA/Catalogs/Twinkle_C/'
fraction = False
flag_slices = ['Unflagged', ]
bins = np.logspace(-3, 4, num=1001)
aspect_ratio = 1
# This parameter tells how many bins to put in the rev-index window
p = 4

RFI = rfipy.RFI(obs, obspath)
xticks = [RFI.UV.Nfreqs * k / 8 for k in range(8)]
xminors = AutoMinorLocator(4)

for m in range(0, len(bins) - 1, p):
    band = {'Unflagged': [bins[m], bins[m + p]], }
    cf.waterfall_catalog(RFI, outpath, band=band, fraction=fraction, bins=bins,
                         flag_slices=flag_slices, xticks=xticks, xminors=xminors,
                         aspect_ratio=aspect_ratio, fit={'Unflagged': False, },
                         write={'Unflagged': False, })

band['Unflagged'] = [5e2, 1e4]
cf.waterfall_catalog(RFI, outpath, band=band, fraction=fraction, bins=bins,
                     flag_slices=flag_slices, xticks=xticks, xminors=xminors,
                     aspect_ratio=aspect_ratio, fit={'Unflagged': False, },
                     write={'Unflagged': False, })

cf.vis_avg_catalog(RFI, outpath, xticks=xticks, xminors=xminors, aspect_ratio=1)
