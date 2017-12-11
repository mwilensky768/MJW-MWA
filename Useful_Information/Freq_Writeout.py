import pyuvdata
import numpy as np

UV = pyuvdata.UVData()

UV.read_uvfits('/Users/mike_e_dubs/MWA/Data/smaller_uvfits/s1061313008.uvfits')

np.save('/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/MWA_Highband_Freq_Array.npy', UV.freq_array[0, :])
