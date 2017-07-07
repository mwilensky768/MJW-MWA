import pyuvdata as pyuv
import numpy as np
import matplotlib.pyplot as plt

UV = pyuv.UVData()
UV.read_uvfits('/Users/mike_e_dubs/python_stuff/uvfits/1061313128.uvfits')
UV.select(times = max(UV.time_array))


for k in range(2):
    D = UV.data_array[:,0,:,0]
    D = np.absolute(D)

    plt.figure(k+1)
    plt.imshow(D,interpolation = 'none')
    plt.ylabel('Baseline #')
    plt.xlabel('Channel #')
    plt.title('Final time Cotter zero check 1061313128 '+['XX','YY'][k])
    plt.colorbar()

plt.show()
