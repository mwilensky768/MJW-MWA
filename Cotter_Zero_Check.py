import pyuvdata as pyuv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

UV = pyuv.UVData()
UV.read_uvfits('/Users/mike_e_dubs/python_stuff/uvfits/1061313128.uvfits')
times = np.ndarray.tolist(UV.time_array)
MAX = max(times)
while MAX in times:
    times.remove(MAX)
MAX = max(times)

UV.select(times = MAX)


for k in range(2):
    D = UV.data_array[:,0,:,k]
    D = np.absolute(D)

    plt.figure(k+1)
    plt.imshow(D,interpolation = 'none', aspect = 384/8128.0, norm = colors.LogNorm())
    plt.ylabel('Baseline #')
    plt.xlabel('Channel #')
    plt.title('2nd-to-last time Cotter zero check 1061313128 '+['XX','YY'][k])
    plt.colorbar()

plt.show()
