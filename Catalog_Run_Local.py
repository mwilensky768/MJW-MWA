import rfipy as rfi

RFI = rfi.RFI()

RFI.rfi_catalog([1061313008, ],
                '/Users/mike_e_dubs/python_stuff/uvfits/1061313008.uvfits',
                '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/')
