import rfipy as rfi
import Catalog_Funcs as cf
import argparse
import glob
import numpy as np
from matplotlib.ticker import FixedLocator, AutoMinorLocator

obslist_path = '/nfs/eor-00/h1/mwilensk/Diffuse_2015_8s_Autos/Diffuse_2015_Good_Pointings_Funky.txt'
pathlist_path = '/nfs/eor-00/h1/mwilensk/Diffuse_2015_8s_Autos/Diffuse_2015_Good_Pointings_Funky_paths.txt'
outpath = '/nfs/eor-00/h1/mwilensk/Diffuse_2015_8s_Autos/Catalogs/Good_Pointings/Funky/Vis_Avg/Amp_First/'
flag_slices = ['All', ]
write = {'Unflagged': False, 'All': False}
writepath = '/nfs/eor-00/h1/mwilensk/S2_Zenith_Calcut_8s_Autos/Catalogs/Ant_Pol/Chirp_Arr/'
bins = np.logspace(-3, 5, num=1001)
catalog_type = 'vis_avg'
plot_type = 'ant-time'
band = {'Unflagged': 'fit', 'All': [2.25e+03, 1e+05]}
auto_remove = True
fit = {'Unflagged': True, 'All': False}
bin_window = [0, 1e+03]
fit_window = [0, 1e+12]
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
output = outpath + str(obs) + '*.png'
output_list = glob.glob(output)

if not output_list:

    RFI = rfi.RFI(str(obs), inpath, auto_remove=auto_remove)
    xticks = [RFI.UV.Nfreqs * k / 6 for k in range(6)]
    xticks.append(RFI.UV.Nfreqs - 1)
    xminors = AutoMinorLocator(4)

    if catalog_type is 'waterfall':
        RFI.rfi_catalog(outpath, write=write, writepath=writepath, bins=bins,
                        band=band, flag_slices=flag_slices, plot_type=plot_type,
                        fit=fit, fit_window=fit_window, bin_window=bin_window,
                        fraction=False)
    elif catalog_type is 'vis_avg':
        cf.vis_avg_catalog(RFI, outpath, band=band[flag_slices[0]], xticks=xticks,
                           flag_slice=flag_slices[0], yminors='auto', xminors=xminors,
                           amp_avg='Amp')
    elif catalog_type is 'temperature':
        RFI.one_d_hist_prepare(flag_slice=flag_slices[2], bins=bins,
                               fit_window=fit_window, bin_window=bin_window,
                               write=write, writepath=writepath)
    elif catalog_type is 'ant-scatter':
        RFI.ant_scatter(outpath, band=band['All'], flag_slice=flag_slices[0])
    elif plot_type is 'ant-pol':
        RFI.ant_pol_catalog(outpath, band=band['All'], clip=clip,
                            write=write['Unflagged'], writepath=writepath)
else:
    print('I already processed obs ' + str(obs))
