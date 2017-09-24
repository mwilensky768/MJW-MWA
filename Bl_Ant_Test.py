import pyuvdata
import numpy as np

UV = pyuvdata.UVData()
UV.read_uvfits('/Users/mike_e_dubs/python_stuff/smaller_uvfits/s1061313008.uvfits')

blt_inds = [k for k in range(self.UV.Blts) if UV.ant_1_array[k] != ]
