import rfipy as rfi

RFI = rfi.RFI(1061313008, '/Users/mike_e_dubs/python_stuff/uvfits/1061313008.uvfits')

RFI.digital_gain_compare('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Dig_Gain_Plots/')
