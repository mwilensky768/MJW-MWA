import rfipy as rfi
import argparse
import os

with open('/nfs/eor-00/h1/mwilensk/RFI_Diagnostic_Diffuse_2015/sidelobe_survey_obsIDs.txt') as f:
    obslist = f.read().split("\n")
with open('/nfs/eor-00/h1/mwilensk/RFI_Diagnostic_Diffuse_2015/sidelobe_survey_obsIDs_paths.txt') as g:
    pathlist = g.read().split("\n")

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
args = parser.parse_args()

if not os.path.exists('/nfs/eor-00/h1/mwilensk/RFI_Diagnostic_Diffuse_2015/' +
                      str(obslist[args.id - 1]) + 'RFI_Diagnostic_All.png'):

    RFI = rfi.RFI()

    RFI.rfi_catalog(obslist[args.id - 1], pathlist[args.id - 1],
                    '/nfs/eor-00/h1/mwilensk/RFI_Diagnostic_Diffuse_2015/',
                    thresh_min=3000, hist_write=True,
                    hist_write_path='/nfs/eor-00/h1/mwilensk/RFI_Diagnostic_Diffuse_2015/Diffuse_2015_Hists/')
else:
    print('I already processed obs ' + str(obslist[args.id - 1]))
