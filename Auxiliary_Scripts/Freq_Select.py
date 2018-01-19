import pyuvdata
import numpy as np

obspath = '/Users/mike_e_dubs/MWA/Data/uvfits/1061313128.uvfits'
obspath_new = '/Users/mike_e_dubs/MWA/Data/smaller_uvfits/1061313128f_180_190.uvfits'

UV = pyuvdata.UVData()
UV.read_uvfits(obspath)

freq_chans = np.where((1.8e8 < UV.freq_array) & (UV.freq_array < 1.9e8))[1]
UV.select(freq_chans=freq_chans)

print(UV.freq_array)

UV.write_uvfits(obspath_new)
