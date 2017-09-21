import rfipy as rfi

RFI = rfi.RFI('1065538888',
              '/Users/mike_e_dubs/python_stuff/uvfits/1065538888.uvfits',
              filetype='uvfits')

RFI.rfi_catalog('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Waterfall_Plots/',
                band=(2000, 10**5), fit=True)
