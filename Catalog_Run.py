import EvenMinusOdd as emo
import argparse

with open('/nfs/eor-00/h1/mwilensk/FHD/obs_list/beardsley_thesis_list.txt') as f:
    obslist = f.read().split("\n")
with open('/nfs/eor-00/h1/mwilensk/FHD/obs_list/beardsley_thesis_list_paths.txt') as g:
    pathlist = g.read().split("\n")

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
args = parser.parse_args()

EMO = emo.EvenMinusOdd(False, True)

EMO.rfi_catalog([obslist[args.id - 1], ], pathlist[args.id - 1],
                '/nfs/eor-00/h1/mwilensk/RFI_Diagnostic_Long_Run/')
