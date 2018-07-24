from pyuvdata import UVData
from rfipy import RFI
import numpy as np

path = '/Users/mike_e_dubs/MWA/Data/smaller_uvfits/s1061313128.uvfits'
obs = 's1061313128'
outpath = '/Users/mike_e_dubs/MWA/Test_Plots/s1061313128_class_transfer'

UV = UVData()
UV.read_uvfits(path)

RFI = RFI(obs, path, outpath, 'uvfits')

RFI.UV = UV

RFI.UV.data_array = np.absolute(np.diff(RFI.UV.data_array.reshape([RFI.UV.Ntimes,
                                                                   RFI.UV.Nbls,
                                                                   RFI.UV.Nspws,
                                                                   RFI.UV.Nfreqs,
                                                                   RFI.UV.Npols]), axis=0))

assert(np.all(RFI.UV.data_array == UV.data_array))
assert(RFI.UV is UV)
