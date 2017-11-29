import rfipy as rfi
import Catalog_Funcs as cf
import argparse
import glob
import numpy as np
from matplotlib.ticker import FixedLocator, AutoMinorLocator

obslist_path = '/nfs/eor-00/h1/mwilensk/S2_Zenith_Calcut_8s_Autos/S2_Zenith_Calcut_8s_Autos_Broadband.txt'
pathlist_path = '/nfs/eor-00/h1/mwilensk/S2_Zenith_Calcut_8s_Autos/S2_Zenith_Calcut_8s_Autos_Broadband_paths.txt'
outpath = '/nfs/eor-00/h1/mwilensk/S2_Zenith_Calcut_8s_Autos/Catalogs/Vis_Avg/Amp_First/Broadband/'
flag_slices = ['All', ]
write = {'Unflagged': False, 'All': False}
writepath = '/nfs/eor-00/h1/mwilensk/S2_Zenith_Calcut_8s_Autos/Catalogs/Ant_Pol/Chirp_Arr/'
bins = np.logspace(-3, 5, num=1001)
fraction = True
catalog_types = ['vis_avg', ]
drill_type = 'time'
band = {'Unflagged': 'fit', 'All': [2.25e+03, 1e+05]}
auto_remove = True
fit = {'Unflagged': True, 'All': False}
bin_window = [0, 1e+03]
fit_window = [0, 1e+12]
clip = True
amp_avg = 'Amp'

with open(obslist_path) as f:
    obslist = f.read().split("\n")
with open(pathlist_path) as g:
    pathlist = g.read().split("\n")

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
args = parser.parse_args()

obs = obslist[args.id - 1]
inpath = pathlist[args.id - 1]
output = outpath + str(obs) + '*.png'
output_list = glob.glob(output)

if not output_list:

    RFI = rfi.RFI(str(obs), inpath, auto_remove=auto_remove)
    xticks = [RFI.UV.Nfreqs * k / 6 for k in range(6)]
    xticks.append(RFI.UV.Nfreqs - 1)
    xminors = AutoMinorLocator(4)

    if 'waterfall' in catalog_types:
        cf.waterfall_catalog(RFI, outpath, write=write, writepath=writepath, bins=bins,
                             band=band, flag_slices=flag_slices, plot_type=plot_type,
                             fit=fit, bin_window=bin_window, fraction=fraction)
    if 'drill' in catalog_types:
        cf.drill_catalog(RFI, outpath, band=band, write=write,
                         writepath=writepath, fit=fit, bins=bins,
                         flag_slices=flag_slices, bin_window=bin_window,
                         xticks=xticks, xminors=xminors, drill_type='time')
    if 'vis_avg' in catalog_types:
        cf.vis_avg_catalog(RFI, outpath, band=band[flag_slices[0]], xticks=xticks,
                           flag_slice=flag_slices[0], yminors='auto', xminors=xminors,
                           amp_avg=amp_avg)
    if 'temperature' in catalog_types:
        RFI.one_d_hist_prepare(flag_slice='Unflagged', bins=bins,
                               bin_window=bin_window, write=True,
                               writepath=writepath)
    if 'ant-scatter' in catalog_types:
        cf.ant_scatter_catalog(RFI, outpath, band['All'], flag_slice=flag_slices[0])
    if 'ant-pol' in catalog types:
        cf.ant_pol_catalog(RFI, outpath, band=band['All'], clip=clip)
else:
    print('I already processed obs ' + str(obs))
