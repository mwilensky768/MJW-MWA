import rfipy as rfi
import argparse
import glob
import numpy as np

# Set these in the beginning every time! Also remember to pick the right type of catalog!

obslist_path = '/nfs/eor-00/h1/mwilensk/S2_Zenith_Calcut_8s_Autos/S2_Zenith_Calcut_8s_Autos_Broadband.txt'
pathlist_path = '/nfs/eor-00/h1/mwilensk/S2_Zenith_Calcut_8s_Autos/S2_Zenith_Calcut_8s_Autos_Broadband_paths.txt'
outpath = '/nfs/eor-00/h1/mwilensk/S2_Zenith_Calcut_8s_Autos/Catalogs/Ant_Pol/'
flag_slices = ['All', ]
write = {'Unflagged': True, 'All': False}
writepath = '/nfs/eor-00/h1/mwilensk/S2_Zenith_Calcut_8s_Autos/Hists/'
bins = np.logspace(-3, 5, num=1001)
catalog_type = ''
plot_type = 'ant-pol'
band = {'Unflagged': 'fit', 'All': [4 * 10**3, 10**5]}
auto_remove = True
fit = {'Unflagged': True, 'All': False}
bin_window = [0, 10**3]
fit_window = [0, 10**12]

with open(obslist_path) as f:
    obslist = f.read().split("\n")
with open(pathlist_path) as g:
    pathlist = g.read().split("\n")

obs_time = [[10, 11], [16, 17], [27, 28], range(2, 11), [22, 23], range(13, 17), 
            [33, 34, 42, 43], [5, 6, 14, 15], [2, 3], [0, 1, 12, 13, 14, 15, 16, 17, 18, 35, 36, 50],
            [29, 30, 44, 45], [6, 7], [8, 9, 23, 24, 33, 34], [3, 4, 11, 12, 34, 35],
            [0, 26, 27], [13, 14], [4, 5, 6, 30, 31], [17, ]]
ant_pol_times = {obslist[k]: obs_time[k] for k in range(len(obs_time))}

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
        RFI.ant_pol_catalog(outpath, ant_pol_times[obs], range(RFI.UV.Nfreqs))
    elif catalog_type is 'Temperature':
        RFI.one_d_hist_prepare(flag_slice=flag_slices[2], bins=bins,
                               fit_window=fit_window, bin_window=bin_window,
                               write=write, writepath=writepath)
else:
    print('I already processed obs ' + str(obs))
