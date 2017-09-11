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

outpaths = glob.glob('/nfs/eor-00/h1/mwilensk/Drill_Plots_Golden_Set/' +
                     str(obslist[args.id - 1]) + '*.png')

if len(outpaths) == 0:

    RFI = rfi.RFI()

    RFI.catalog_drill(obslist[args.id - 1], pathlist[args.id - 1],
                      '/nfs/eor-00/h1/mwilensk/Drill_Plots_Golden_Set/', 'ant-freq')
else:
    print('I already processed obs ' + str(obslist[args.id - 1]))
