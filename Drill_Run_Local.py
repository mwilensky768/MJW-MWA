import rfipy as rfi

RFI = rfi.RFI('1061318864', '/Users/mike_e_dubs/python_stuff/uvfits/1061318864.uvfits',
              bad_time_indices=[0, 53, 54, 55])

RFI.catalog_drill('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Drill_Plots/',
                  plot_type='ant-time', band=(1500, 10**5))
