import pyuvdata
import numpy as np

UV = pyuvdata.UVData()

for t in [11, ]:
    UV.read_uvfits('/Users/mike_e_dubs/MWA/data/smaller_uvfits/1061313128_t%i_no_flag_mod.uvfits' % (t))
    ind = np.where(UV.nsample_array == 0)
    for m in range(len(ind[0])):
        UV.nsample_array[ind[0][m], ind[1][m], ind[2][m], ind[3][m]] = \
            UV.nsample_array[0, ind[1][m], ind[2][m] % 16, ind[3][m]]

    UV.flag_array[:, :, :176, :] = 1
    UV.flag_array[:, :, 256:, :] = 1
    UV.flag_array[:, :, 176:256, :] = 0

    UV.write_uvfits('/Users/mike_e_dubs/MWA/data/smaller_uvfits/1061313128_t%i_flag_mod.uvfits' % (t))
