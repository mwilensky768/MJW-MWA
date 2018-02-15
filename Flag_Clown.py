import pyuvdata
import numpy as np

Cal = pyuvdata.UVCal()
Cal.read_fhd_cal('/Users/mike_e_dubs/MWA/FHD/fhd_mjw_Aug23_Jan2018/calibration/1061313128_f181.2_f187.5_t30_t36_cal.sav',
                 '/Users/mike_e_dubs/MWA/FHD/fhd_mjw_Aug23_Jan2018/metadata/1061313128_f181.2_f187.5_t18_t24_obs.sav')

print(Cal.flag_array.shape)
