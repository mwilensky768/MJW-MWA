import pyuvdata as pyuv
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

UV = pyuv.UVData()

UV.read_uvfits('/Users/mike_e_dubs/python_stuff/uvfits/1061313008.uvfits')

baseline_time_indices = []

for m in range(UV.Nblts):
    if UV.ant_1_array[m] == UV.ant_2_array[m]:
        baseline_time_indices.append(m)

UV.select(blt_inds=baseline_time_indices)

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()

Hxy = np.ma.masked_equal(np.absolute(UV.data_array[:, 0, :, 2]), 0)
Hyx = np.ma.masked_equal(np.absolute(UV.data_array[:, 0, :, 3]), 0)
cmap = cm.cool
cmap.set_bad(color='white')

caxXY = ax1.imshow(Hxy, cmap=cmap, vmin=np.amin(Hxy), vmax=np.amax(Hxy))
caxYX = ax2.imshow(Hyx, cmap=cmap, vmin=np.amin(Hyx), vmax=np.amax(Hyx))

cbarXY = fig1.colorbar(caxXY, ax=ax1)
cbarYX = fig2.colorbar(caxYX, ax=ax2)

ax1.set_title('XY Autocorrelations 0 Check')
ax2.set_title('YX Autocorrelations 0 Check')

fig1.savefig('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/XY_Check.png')
fig2.savefig('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/YX_Check.png')

plt.show()
