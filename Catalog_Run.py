import rfipy as rfi
import argparse
import glob
import numpy as np

# Set these in the beginning every time! Also remember to pick the right type of catalog!

obslist_path = '/nfs/eor-00/h1/mwilensk/Diffuse_2015_8s_Autos/diffuse_survey_good_pointings.txt'
pathlist_path = '/nfs/eor-00/h1/mwilensk/Diffuse_2015_8s_Autos/diffuse_survey_good_pointings_paths.txt'
outpath = '/nfs/eor-00/h1/mwilensk/Diffuse_2015_8s_Autos/Catalogs/Good_Pointings/Freq_Time/'
flag_slices = ['All', 'Unflagged']
write = {'Unflagged': True, 'All': False}
writepath = '/nfs/eor-00/h1/mwilensk/Diffuse_2015_8s_Autos/Temperatures/Good_Pointings/Hists/'
bins = np.logspace(-3, 5, num=1001)
catalog_type = 'waterfall'
plot_type = 'freq-time'
band = {'Unflagged': 'fit', 'All': [2.25e+03, 1e+05]}
auto_remove = True
fit = {'Unflagged': True, 'All': False}
bin_window = [0, 1e+04]
fit_window = [0, 1e+12]
clip = True
fraction = True

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
                        fraction=fraction)
    elif catalog_type is 'vis_avg':
        RFI.vis_avg_catalog(outpath, band=band[flag_slices[0]], flag_slice=flag_slices[0])
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
