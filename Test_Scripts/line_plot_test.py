import rfipy

RFI = rfipy.RFI('1061313008',
                '/Users/mike_e_dubs/MWA/Data/smaller_uvfits/s1061313008.uvfits')

RFI.vis_avg_catalog('/Users/mike_e_dubs/MWA/Misc/Misc_Checks/',
                    band=[1.5e+03, 1e+05],
                    flag_slice='All')
