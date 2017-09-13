import rfipy as rfi
import argparse
import os

with open('/nfs/eor-00/h1/mwilensk/RunTexts/Golden_Set_OBSIDS.txt') as f:
    obslist = f.read().split("\n")
with open('/nfs/eor-00/h1/mwilensk/RunTexts/Golden_Set_OBSIDS_paths.txt') as g:
    pathlist = g.read().split("\n")

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
args = parser.parse_args()

if not os.path.exists('/nfs/eor-00/h1/mwilensk/RFI_Diagnostic_Golden_Set/Digital_Gain_Comparison/' +
                      str(obslist[args.id - 1]) + '_DGC.png'):

    RFI = rfi.RFI()

    RFI.digital_gain_compare(obslist[args.id - 1], pathlist[args.id - 1],
                             '/nfs/eor-00/h1/mwilensk/RFI_Diagnostic_Diffuse_2015/')
else:
    print('I already processed obs ' + str(obslist[args.id - 1]))
