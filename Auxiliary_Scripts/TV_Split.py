import pyuvdata
import numpy as np

UV = pyuvdata.UVData()
UV.read_uvfits('/Users/mike_e_dubs/MWA/data/uvfits/1061313128.uvfits')
# UV.select(freq_chans=range(176, 256))

UV.flag_array[:, :, :176, :] = 1
UV.flag_array[:, :, 256:, :] = 1
UV.flag_array[:, :, range(176, 256), :] = 0

times1 = UV.time_array[18 * UV.Nbls:25 * UV.Nbls:UV.Nbls]
times2 = UV.time_array[30 * UV.Nbls:37 * UV.Nbls:UV.Nbls]

UV1 = UV.select(times=times1, inplace=False)
UV2 = UV.select(times=times2, inplace=False)

UV1.write_uvfits('/Users/mike_e_dubs/MWA/data/smaller_uvfits/1061313128_t18_t24_flag_mod.uvfits')
UV2.write_uvfits('/Users/mike_e_dubs/MWA/data/smaller_uvfits/1061313128_t30_t36_flag_mod.uvfits')
