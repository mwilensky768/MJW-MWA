import rfipy as rfi
import argparse
import glob
import numpy as np

# Set these in the beginning every time! Also remember to pick the right type of catalog!

obslist_path = '/nfs/eor-00/h1/mwilensk/Long_Run_8s_Autos/Long_Run_8s_Autos_Departure_OBSIDS.txt'
pathlist_path = '/nfs/eor-00/h1/mwilensk/Long_Run_8s_Autos/Long_Run_8s_Autos_Departure_OBSIDS_paths.txt'
outpath = '/nfs/eor-00/h1/mwilensk/Long_Run_8s_Autos/Long_Run_8s_Autos_Departure_Waterfall_Plots/'
hist_write_path = '/nfs/eor-00/h1/mwilensk/Long_Run_8s_Autos/Long_Run_8s_Autos_Hists/'
bins = np.logspace(-3, 5, num=1001)
catalog_type = 'waterfall'
flag_slices = ['Or', 'All']
band = [400, 10**5]

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

    RFI = rfi.RFI(str(obs), inpath, auto_remove=True)

    if catalog_type == 'waterfall':
        RFI.rfi_catalog(outpath, hist_write=False, hist_write_path=hist_write_path,
                        bins=bins, band=band, flag_slices=flag_slices)
    elif catalog_type == 'drill':
        RFI.catalog_drill(outpath, plot_type='ant-time', band=(2000, 10**5),
                          bins=bins)
    elif catalog_type == 'ant-pol':
        RFI.ant_pol_catalog(outpath, range(RFI.UV.Ntimes), [162, ])
else:
    print('I already processed obs ' + str(obs))
