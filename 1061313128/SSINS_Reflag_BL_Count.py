import numpy as np
from SSINS import plot_lib
import matplotlib.pyplot as plt
from pyuvdata import UVData

UV = UVData()
UV.read('/Users/mike_e_dubs/MWA/Data/uvfits/1061313128_SSINS_reflag.uvfits',
        polarizations=[-5])
print(UV.flag_array.shape)
UV.flag_array = UV.flag_array.reshape([UV.Ntimes, UV.Nbls, UV.Nspws, UV.Nfreqs, UV.Npols])
print(UV.flag_array.shape)
BL_Count = np.count_nonzero(UV.flag_array, axis=1)
np.save('/Users/mike_e_dubs/MWA/PS/SSINS_Reflag_BL_Count.npy', BL_Count)

fig, ax = plt.subplots(figsize=(14, 8))
plot_lib.image_plot(fig, ax, BL_Count[:, 0, :, 0], xlabel='Channel #',
                    cbar_label='# Baselines')
fig.savefig('/Users/mike_e_dubs/MWA/PS/SSINS_Reflag_BL_Count.png')
