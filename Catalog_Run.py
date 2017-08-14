import EvenMinusOdd as emo
import argparse
import os

with open('/nfs/eor-00/h1/mwilensk/FHD/obs_list/beardsley_thesis_list.txt') as f:
    obslist = f.read().split("\n")
with open('/nfs/eor-00/h1/mwilensk/FHD/obs_list/beardsley_thesis_list_paths.txt') as g:
    pathlist = g.read().split("\n")

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
args = parser.parse_args()

if not os.path.exists('/nfs/eor-00/h1/mwilensk/RFI_Diagnostic_Long_Run/' +
                      str(obslist[args.id - 1]) + '_RFI_Diagnostic_All.png'):

    EMO = emo.EvenMinusOdd(False, True)

    EMO.rfi_catalog([obslist[args.id - 1], ], pathlist[args.id - 1],
                    '/nfs/eor-00/h1/mwilensk/RFI_Diagnostic_Long_Run/')
else:
    print('I already processed obs ' + str(obslist[args.id - 1]))
