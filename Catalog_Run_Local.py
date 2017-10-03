import rfipy as rfi
import numpy as np

# bad_blt_inds = np.load('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/bad_blt_inds.npy')
# bad_blt_inds = bad_blt_inds.tolist()

obs = '1061313008'

RFI = rfi.RFI(obs,
              '/Users/mike_e_dubs/python_stuff/smaller_uvfits/s1061313008.uvfits',
              filetype='uvfits', auto_remove=True)

RFI.rfi_catalog('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Waterfall_Plots/MWA/',
                band=(400, 10**5), fit=False)
