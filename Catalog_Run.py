import rfipy as rfi
import argparse
import glob
import numpy as np

# Set these in the beginning every time! Also remember to pick the right type of catalog!

obslist_path = '/nfs/eor-00/h1/mwilensk/Golden_Set/Golden_Set_Narrowband_OBSIDS.txt'
pathlist_path = '/nfs/eor-00/h1/mwilensk/Golden_Set/Golden_Set_Narrowband_OBSIDS_paths.txt'
outpath = '/nfs/eor-00/h1/mwilensk/Golden_Set/Golden_Set_Drill_Plots/Golden_Set_Drill_Plots_Narrowband/'
flag_slices = ['All', ]
write = {'Unflagged': True, 'All': False}
writepath = '/nfs/eor-00/h1/mwilensk/S2_Zenith_Calcut_8s_Autos/Hists/'
bins = np.logspace(-3, 5, num=1001)
catalog_type = 'waterfall'
plot_type = 'ant-time'
band = {'Unflagged': 'fit', 'All': [1.5 * 10**3, 10**5]}
auto_remove = True
fit = {'Unflagged': True, 'All': False}
bin_window = [0, 10**3]
fit_window = [0, 10**12]

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

    if catalog_type is 'waterfall':
        RFI.rfi_catalog(outpath, write=write, writepath=writepath, bins=bins,
                        band=band, flag_slices=flag_slices, plot_type=plot_type,
                        fit=fit, fit_window=fit_window, bin_window=bin_window,
                        fraction=False)
    elif plot_type is 'ant-pol':
        RFI.ant_pol_catalog(outpath, band=band['All'])
    elif catalog_type is 'Temperature':
        RFI.one_d_hist_prepare(flag_slice=flag_slices[2], bins=bins,
                               fit_window=fit_window, bin_window=bin_window,
                               write=write, writepath=writepath)
else:
    print('I already processed obs ' + str(obs))
