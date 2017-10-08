import rfipy as rfi
import argparse
import glob
import numpy as np

obslist_path = '/nfs/eor-00/h1/mwilensk/Long_Run_8s_Autos/Long_Run_8s_Autos_OBSIDS.txt'
pathlist_path = '/nfs/eor-00/h1/mwilensk/Long_Run_8s_Autos/Long_Run_8s_Autos_OBSIDS_paths.txt'
write = True
temp_write = True
writepath = '/nfs/eor-00/h1/mwilensk/Long_Run_8s_Autos/Temperatures/'
bins = np.logspace(-3, 3.3, num=1001)
fit = True
fit_window = [0, 10**12]
cutlist_path = '/nfs/eor-00/h1/mwilensk/Long_Run_8s_Autos/Long_Run_8s_Autos_Funky_OBSIDS.txt'
cutpathlist_path = '/nfs/eor-00/h1/mwilensk/Long_Run_8s_Autos/Long_Run_8s_Funky_Funky_OBSIDS_paths.txt'
filetype = 'uvfits'
flag_slice = 'All'

with open(obslist_path) as f:
    obslist = f.read().split("\n")
with open(pathlist_path) as g:
    pathlist = g.read().split("\n")
with open(cutlist_path) as h:
    cutlist = h.read().split("\n")
with open(cutpathlist_path) as k:
    cutpathlist = k.read().split("\n")

for item in cutlist:
    obslist.remove(item)
for item in cutpathlist:
    pathlist.remove(item)

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
args = parser.parse_args()

obs = obslist[args.id - 1]
inpath = pathlist[args.id - 1]
output = writepath + str(obs) + '*.npy'
output_list = glob.glob(output)

if not output_list:

    RFI = rfi.RFI(obs, inpath, filetype=filetype)

    RFI.one_d_hist_prepare(bins=bins, fit=fit, fit_window=fit_window, write=write,
                           writepath=writepath, temp_write=temp_write,
                           flag_slice=flag_slice)
else:
    print('I already processed obs ' + obs)
