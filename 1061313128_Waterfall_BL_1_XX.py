import pyuvdata as pyuv
import numpy as np
import matplotlib.pyplot as plt

UV = pyuv.UVData()

UV.read_uvfits('/Users/mike_e_dubs/python_stuff/uvfits/1061313128.uvfits')

UV.select(blt_inds = [1+8128*m for m in range(56)])


plt.figure()

plt.imshow(np.absolute(UV.data_array[:,0,:,0]))
plt.xlabel('time (2s)')
plt.ylabel('frequency (channel #)')
plt.title('1061313128 Baseline 1 XX Vis. Waterfall')
plt.xticks([2*k for k in range(28)])
plt.yticks([16*k for k in range(24)])
plt.colorbar()

plt.show()
