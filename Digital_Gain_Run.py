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

obs = obslist[args.id - 1]
inpath = pathlist[agrs.id - 1]
outpath = '/nfs/eor-00/h1/mwilensk/RFI_Diagnostic_Golden_Set/Digital_Gain_Comparison/'

if not os.path.exists(outpath + str(obs) + '_Normed_DGC.png'):

    RFI = rfi.RFI(obs, inpath)

    RFI.digital_gain_compare(outpath)
else:
    print('I already processed obs ' + str(obs))
