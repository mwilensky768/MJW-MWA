import rfipy as rfi
import numpy as np
import matplotlib.pyplot as plt
import glob

pathlist = glob.glob('/Users/mike_e_dubs/python_stuff/miriad/temp_HERA_data/*.uvc')
outpath = '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Waterfall_Plots/'

for path in pathlist:
    start = path.find('zen.')
    end = path.find('.uvc')
    obs = path[start:end]

    RFI = rfi.RFI(obs, path, filetype='miriad')

    RFI.rfi_catalog(outpath, band=(0.5, 10**4))
