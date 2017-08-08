import EvenMinusOdd as emo
import argparse

with open('/nfs/eor-00/h1/mwilensk/FHD/obs_list/Aug23.txt') as f:
    obslist = f.read().split("\n")

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
args = parser.parse_args()

EMO = emo.EvenMinusOdd(False, False)

EMO.rfi_catalog([obslist[args.id - 1], ], '/nfs/eor-11/r1/EoRuvfits/jd2456528v4_1/',
                '/nfs/eor-00/h1/mwilensk/RFI_Diagnostic/')
