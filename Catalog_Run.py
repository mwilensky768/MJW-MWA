import rfipy as rfi
import Catalog_Funcs as cf
import argparse
import glob
import numpy as np
from matplotlib.ticker import FixedLocator, AutoMinorLocator

"""Input/Output keywords"""

catalog_types = ['vis_avg', ]
obslist_path = '/nfs/eor-00/h1/mwilensk/Diffuse_2015_10s_Autos/Diffuse_2015_GP_10s_Autos_RFI_Free.txt'
pathlist_path = '/nfs/eor-00/h1/mwilensk/Diffuse_2015_10s_Autos/Diffuse_2015_GP_10s_Autos_RFI_Free_paths.txt'
outpath = {'waterfall': '/nfs/eor-00/h1/mwilensk/Diffuse_2015_10s_Autos/catalogs/freq_time/',
           'vis_avg': '/nfs/eor-00/h1/mwilensk/Diffuse_2015_10s_Autos/catalogs/vis_avg/waterfall/amp_first/'}

"""Object Keywords"""

bad_time_indices = [0, -4, -3, -2, -1]
auto_remove = True

"""Misc. Keywords"""

flag_slices = ['All', 'Unflagged']
write = {'Unflagged': False, 'All': False}
writepath = '/nfs/eor-00/h1/mwilensk/S2_Zenith_Calcut_8s_Autos/Catalogs/Ant_Pol/Chirp_Arr/'
bins = np.logspace(-3, 5, num=1001)
band = {'Unflagged': 'fit', 'All': [2e+03, 1e+05]}
fit = {'Unflagged': True, 'All': False}
bin_window = [0, 2e+03]

"""Waterfall Keywords"""

fraction = True

"""Drill Keywords"""

drill_type = 'time'

"""Vis_Avg Keywords"""

amp_avg = 'Amp'
plot_type = 'waterfall'
vis_avg_write = True
vis_avg_writepath = '/nfs/eor-00/h1/mwilensk/Diffuse_2015_10s_Autos/catalogs/vis_avg/waterfall/amp_first/arrays/'

"""Ant_Pol Keywords"""

clip = True

with open(obslist_path) as f:
    obslist = f.read().split("\n")
with open(pathlist_path) as g:
    pathlist = g.read().split("\n")

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
args = parser.parse_args()

obs = obslist[args.id - 1]
inpath = pathlist[args.id - 1]
output = '%s%s*.png' % (outpath[catalog_types[0]], str(obs))
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
    if 'vis_avg' in catalog_types:
        cf.vis_avg_catalog(RFI, outpath['vis_avg'], band=band[flag_slices[0]], xticks=xticks,
                           flag_slice=flag_slices[0], yminors='auto', xminors=xminors,
                           amp_avg=amp_avg, plot_type=plot_type, write=vis_avg_write,
                           writepath=vis_avg_writepath)
    if 'temperature' in catalog_types:
        RFI.one_d_hist_prepare(flag_slice='Unflagged', bins=bins,
                               bin_window=bin_window, write=True,
                               writepath=writepath)
    if 'ant_scatter' in catalog_types:
        cf.ant_scatter_catalog(RFI, outpath, band['All'], flag_slice=flag_slices[0])
    if 'ant_pol' in catalog_types:
        cf.ant_pol_catalog(RFI, outpath, band=band['All'], clip=clip)
else:
    print('I already processed obs ' + str(obs))
