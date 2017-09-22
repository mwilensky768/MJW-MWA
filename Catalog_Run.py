import rfipy as rfi
import argparse
import glob

# Set these in the beginning every time! Also remember to pick the right type of catalog!

obslist_path = '/nfs/eor-00/h1/mwilensk/Golden_Set/Golden_Set_Narrowband_OBSIDS.txt'
pathlist_path = '/nfs/eor-00/h1/mwilensk/Golden_Set/Golden_Set_Narrowband_OBSIDS_paths.txt'
outpath = '/nfs/eor-00/h1/mwilensk/Golden_Set/Golden_Set_Drill_Plots/Golden_Set_Drill_Plots_Narrowband/'
catalog_type = 'drill'

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

bad_time_indices = [0, 53, 54, 55]

if not output_list:

    RFI = rfi.RFI(str(obs), inpath, bad_time_indices=bad_time_indices)

    if catalog_type == 'waterfall':
        RFI.rfi_catalog(outpath, hist_write=True,
                        hist_write_path='/nfs/eor-00/h1/mwilensk/RFI_Diagnostic_Diffuse_2015/Diffuse_2015_Hists/')
    elif catalog_type == 'drill':
        RFI.catalog_drill(outpath, plot_type='ant-time')
    elif catalog_type == 'ant-pol':
        RFI.ant_pol_catalog(outpath, times=[], freqs=[])
else:
    print('I already processed obs ' + str(obs))
