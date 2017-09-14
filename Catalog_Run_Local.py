import rfipy as rfi

RFI = rfi.RFI(1061313008, '/Users/mike_e_dubs/python_stuff/smaller_uvfits/s1061313008.uvfits')

RFI.rfi_catalog('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/')
