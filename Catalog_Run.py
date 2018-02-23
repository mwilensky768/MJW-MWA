import rfipy as rfi
import Catalog_Funcs as cf
import argparse
import glob
import numpy as np
from matplotlib.ticker import FixedLocator, AutoMinorLocator

"""Input/Output keywords"""

catalog_types = ['INS', 'flag', 'waterfall']
obslist_path = '/nfs/eor-00/h1/mwilensk/P2_Bad_Obs/badobs_list_wenyang.txt'
pathlist_path = '/nfs/eor-00/h1/mwilensk/P2_Bad_Obs/badobs_list_wenyang_paths.txt'
outpath = {'waterfall': '/nfs/eor-00/h1/mwilensk/P2_Bad_Obs/Catalogs/Freq_Time/',
           'INS': '/nfs/eor-00/h1/mwilensk/P2_Bad_Obs/Catalogs/INS/',
           'flag': '/nfs/eor-00/h1/mwilensk/P2_Bad_Obs/Catalogs/Flags/'}

"""Object Keywords"""

bad_time_indices = [0, -1]
auto_remove = True

"""Misc. Keywords"""

flag_slices = ['All', 'Unflagged']
write = {'Unflagged': False, 'All': False}
writepath = '/nfs/eor-00/h1/mwilensk/Golden_Set_8s_Autos/Temperatures/Vis_Var/All/'
bins = 'auto'
band = {'Unflagged': 'fit', 'All': [2e+03, 1e+05], 'Flagged': [0, 1e6]}
fit = {'Unflagged': True, 'All': False}
bin_window = [0, 2e+03]

"""Waterfall Keywords"""

fraction = True

"""Drill Keywords"""

drill_type = 'time'

"""INS Keywords"""

amp_avg = 'Amp'
vis_avg_write = False
vis_avg_writepath = '/nfs/eor-00/h1/mwilensk/Golden_Set_8s_Autos/Temperatures/Vis_Avg/All/'
invalid_mask = False

"""Ant_Pol Keywords"""

clip = True

with open(obslist_path) as f:
    obslist = f.read().split("\n")
with open(pathlist_path) as g:
    pathlist = g.read().split("\n")

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
args = parser.parse_args()

"""Filesystem Stuff"""

obs = obslist[args.id - 1]
inpath = pathlist[args.id - 1]
output = '%s%s*.png' % ('/nfs/eor-00/h1/mwilensk/P2_Bad_Obs/Catalogs/Flags/', str(obs))
output_list = glob.glob(output)

if not output_list:

    RFI = rfi.RFI(str(obs), inpath, auto_remove=auto_remove,
                  bad_time_indices=bad_time_indices)
    xticks = [RFI.UV.Nfreqs * k / 6 for k in range(6)]
    xticks.append(RFI.UV.Nfreqs - 1)
    xminors = AutoMinorLocator(4)

    if 'waterfall' in catalog_types:
        cf.waterfall_catalog(RFI, outpath['waterfall'], write=write,
                             writepath=writepath, bins=bins, band=band,
                             flag_slices=flag_slices, fit=fit,
                             bin_window=bin_window, fraction=fraction,
                             xticks=xticks, xminors=xminors)
    if 'drill' in catalog_types:
        cf.drill_catalog(RFI, outpath, band=band, write=write,
                         writepath=writepath, fit=fit, bins=bins,
                         flag_slices=flag_slices, bin_window=bin_window,
                         xticks=xticks, xminors=xminors, drill_type='time')
    if 'INS' in catalog_types:
        cf.INS_catalog(RFI, outpath['INS'], xticks=xticks,
                       flag_slices=flag_slices, yminors='auto',
                       xminors=xminors, amp_avg=amp_avg, write=vis_avg_write,
                       writepath=vis_avg_writepath, invalid_mask=invalid_mask)
    if 'temperature' in catalog_types:
        RFI.one_d_hist_prepare(flag_slice='All', bins=bins, fit=True,
                               bin_window=bin_window, write=True,
                               writepath=writepath)
    if 'ant_scatter' in catalog_types:
        cf.ant_scatter_catalog(RFI, outpath, band['All'], flag_slice=flag_slices[0])
    if 'ant_pol' in catalog_types:
        cf.ant_pol_catalog(RFI, outpath, band=band['All'], clip=clip)
    if 'flag' in catalog_types:
        cf.flag_catalog(RFI, outpath['flag'], xticks=xticks, xminors=xminors)
else:
    print('I already processed obs ' + str(obs))
