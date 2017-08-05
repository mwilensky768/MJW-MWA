import EvenMinusOdd as emo

with open('/nfs/eor-00/h1/mwilensk/FHD/obs_list/Aug23.txt') as f:
    obslist = f.read().split("\n")

EMO = emo.EvenMinusOdd(False, False)

EMO.rfi_catalog(obslist[40:50], '/nfs/eor-11/r1/EoRuvfits/jd2456528v4_1/',
                '/nfs/eor-00/h1/mwilensk/RFI_Diagnostic/')
