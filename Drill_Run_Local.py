import rfipy as rfi

RFI = rfi.RFI(1061313008, '/Users/mike_e_dubs/python_stuff/uvfits/1061313008.uvfits',)

RFI.catalog_drill('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Drill_Plots/', 'ant-freq')
