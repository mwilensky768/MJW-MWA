import pyuvdata
import numpy as np

UV = pyuvdata.UVData()

UV.read_uvfits('/Users/mike_e_dubs/MWA/Data/uvfits/1061313008.uvfits')
np.save('/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/MWA_ant_pos.npy',
        UV.antenna_positions)
