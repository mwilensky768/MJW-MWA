import rfipy as rfi
import numpy as np

# bad_blt_inds = np.load('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/bad_blt_inds.npy')
# bad_blt_inds = bad_blt_inds.tolist()

obslist = ['1061313008', '1061313128', '1065538888', '1061318864']

for obs in obslist:
    RFI = rfi.RFI('1061318864',
                  '/Users/mike_e_dubs/python_stuff/uvfits/1061318864.uvfits',
                  filetype='uvfits', auto_remove=True, )

RFI.rfi_catalog('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Waterfall_Plots/MWA/',
                band=(10**3, 10**5), fit=False)
