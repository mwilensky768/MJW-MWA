import rfipy as rfi
import argparse
import glob
import numpy as np

# Set these in the beginning every time! Also remember to pick the right type of catalog!

obslist_path = '/nfs/eor-00/h1/mwilensk/Long_Run_8s_Autos/Long_Run_8s_Autos_Departure_OBSIDS.txt'
pathlist_path = '/nfs/eor-00/h1/mwilensk/Long_Run_8s_Autos/Long_Run_8s_Autos_Departure_OBSIDS_paths.txt'
outpath = '/nfs/eor-00/h1/mwilensk/Long_Run_8s_Autos/Waterfall_Plots/Departure/'
hist_write = False
hist_write_path = '/nfs/eor-00/h1/mwilensk/Long_Run_8s_Autos/Long_Run_8s_Autos_Hists/'
bins = np.logspace(-3, 5, num=1001)
catalog_type = 'waterfall'
plot_type = 'ant-time'
flag_slices = ['Unflagged', 'Flagged', 'All']
band = [10**3, 10**5]
auto_remove = True

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

    if catalog_type == 'waterfall':
        RFI.rfi_catalog(outpath, hist_write=hist_write, hist_write_path=hist_write_path,
                        bins=bins, band=band, flag_slices=flag_slices)
    elif catalog_type == 'drill':
        RFI.catalog_drill(outpath, plot_type='ant-time', band=band,
                          bins=bins, flag_slices=flag_slices)
    elif catalog_type == 'ant-pol':
        RFI.ant_pol_catalog(outpath, range(RFI.UV.Ntimes), [162, ])
else:
    print('I already processed obs ' + str(obs))
