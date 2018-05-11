import pyuvdata
import numpy as np

UV = pyuvdata.UVData()
UV.read_uvfits('/Users/mike_e_dubs/MWA/data/uvfits/1061313128.uvfits', read_data=False)
# UV.select(freq_chans=range(176, 256))

#UV.flag_array[:, :, :176, :] = 1
#UV.flag_array[:, :, 256:, :] = 1
#UV.flag_array[:, :, 176:256, :] = 0

times = [UV.time_array[t * UV.Nbls:(t + 1) * UV.Nbls:UV.Nbls] for t in [11, ]]
UV.read_uvfits_data('/Users/mike_e_dubs/MWA/data/uvfits/1061313128.uvfits', times=times)
for k, t in enumerate([11, ]):
    # times1 = UV.time_array[t * UV.Nbls:(t + 1) * UV.Nbls:UV.Nbls]
# times2 = UV.time_array[30 * UV.Nbls:37 * UV.Nbls:UV.Nbls]

    UV1 = UV.select(times=[times[k], ], inplace=False)
#UV2 = UV.select(times=times2, inplace=False)

    UV1.write_uvfits('/Users/mike_e_dubs/MWA/data/smaller_uvfits/1061313128_t%i_no_flag_mod.uvfits' % (t))
#UV2.write_uvfits('/Users/mike_e_dubs/MWA/data/smaller_uvfits/1061313128_t30_t36_no_flag_mod.uvfits')
