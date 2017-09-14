import rfipy as rfi
import argparse
import os
import glob

with open('/nfs/eor-00/h1/mwilensk/RunTexts/Broadband_OBSIDS.txt') as f:
    obslist = f.read().split("\n")
with open('/nfs/eor-00/h1/mwilensk/RunTexts/Broadband_OBSIDS_paths.txt') as g:
    pathlist = g.read().split("\n")

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
args = parser.parse_args()

obs = obslist[args.id - 1]
inpath = pathlist[args.id - 1]
outpath = '/nfs/eor-00/h1/mwilensk/Drill_Plots_Golden_Set/'

outpaths = glob.glob('/nfs/eor-00/h1/mwilensk/Drill_Plots_Golden_Set/' +
                     str(obslist[args.id - 1]) + '*.png')

if len(outpaths) == 0:

    RFI = rfi.RFI(obs, inpath)

    RFI.catalog_drill(outpath)
else:
    print('I already processed obs ' + str(obslist[args.id - 1]))
