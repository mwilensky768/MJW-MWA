from pyuvdata import UVData
import numpy as np

fp = '/Users/mike_e_dubs/MWA/Data/uvfits/1061313128.uvfits'

UV = UVData()
UV.read_uvfits(fp, read_data=False)
blt_inds_auto = np.where(UV.ant_1_array == UV.ant_2_array)
if len(blt_inds_auto[0]) == 0:
    print('There are no autocorrelations in this uvfits file to begin with!')

UV.read_uvfits_data(fp, ant_str='cross')
blt_inds_auto = np.where(UV.ant_1_array == UV.ant_2_array)
print('Number of pols is %i' % (UV.Npols))
if len(blt_inds_auto[0]) == 0:
    print('No autocorrelations are present after read selection!')
else:
    print('Autocorrelations are still present despite read selection!')
