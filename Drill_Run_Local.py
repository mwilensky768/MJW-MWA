import rfipy as rfi

RFI = rfi.RFI()

RFI.catalog_drill(1061313008, '/Users/mike_e_dubs/python_stuff/uvfits/',
                  '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Drill_Plots/', 'ant-freq')
