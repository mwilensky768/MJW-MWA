from pyuvdata import UVData
import numpy as np

UV = UVData()
UV.read_uvfits('/data4/mwilensky/uvfits/1061313128_t18_t24_no_flag_mod.uvfits')

# Unflag some middle frequencies and flag all other frequencies
UV.flag_array[:, :, 176:256, :] = 0
UV.flag_array[:, :, :176, :] = 1
UV.flag_array[:, :, 256:, :] = 1

# Check that this assignment worked
assert(not np.any(UV.flag_array[:, :, 176:256, :]))
assert(np.all(UV.flag_array[:, :, :176, :]) and np.all(UV.flag_array[:, :, 256:, :]))

UV.write_uvfits('/Users/mike_e_dubs/MWA/Data/smaller_uvfits/1061313128_t18_t24_flag_mod_write_test.uvfits')
UV2 = UVData()
UV2.read_uvfits('/Users/mike_e_dubs/MWA/Data/smaller_uvfits/1061313128_t18_t24_flag_mod_write_test.uvfits')

# Perform same check as above to see if flag array saved
assert(not np.any(UV2.flag_array[:, :, 176:256, :]))
assert(np.all(UV2.flag_array[:, :, :176, :]) and np.all(UV2.flag_array[:, :, 256:, :]))
