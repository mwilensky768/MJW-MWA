import rfipy as rfi
import argparse
import glob

# Set these in the beginning every time! Also remember to pick the right type of catalog!

obslist_path = '/nfs/eor-00/h1/mwilensk/RFI_Diagnostic_Diffuse_2015/sidelobe_survey_obsIDs.txt'
pathlist_path = '/nfs/eor-00/h1/mwilensk/RFI_Diagnostic_Diffuse_2015/sidelobe_survey_obsIDs_paths.txt'
outpath = '/nfs/eor-00/h1/mwilensk/RFI_Diagnostic_Diffuse_2015/'

with open(obslist_path) as f:
    obslist = f.read().split("\n")
with open(pathlist_path) as g:
    pathlist = g.read().split("\n")

parser = argparse.ArgumentParser()
parser.add_argument("id", type=int)
args = parser.parse_args()

obs = obslist[args.id - 1]
inpath = pathlist[args.id - 1]
output = outpath + str(obs) + '*.png'
output_list = glob.glob(output)

if not output_list:

    RFI = rfi.RFI(obs, inpath)

    RFI.rfi_catalog(outpath, hist_write=True,
                    hist_write_path='/nfs/eor-00/h1/mwilensk/RFI_Diagnostic_Diffuse_2015/Diffuse_2015_Hists/')
else:
    print('I already processed obs ' + str(obs))
