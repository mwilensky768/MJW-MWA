import pyuvdata as pyuv
import matplotlib.pyplot as plt
import numpy as np

UV = pyuv.UVData()
UV.read_uvfits('/Users/mike_e_dubs/python_stuff/uvfits/1061313008.uvfits')

y = UV.ant_1_array[0:UV.Nbls]
x = UV.ant_2_array[0:UV.Nbls]

W = np.where(x == y)

B = [UV.baseline_to_antnums(UV.baseline_array[x]) for x in W[0]]

print('ant_1_array[k] == ant_2_array[k] for the following indices: ')
print(W[0])

print('Here are the corresponding Antenna pair tuples:')
print(B)

fig, ax = plt.subplots()

ax.scatter(x, y)
ax.set_xlabel('Antenna 2')
ax.set_ylabel('Antenna 1')

ax.plot(x, x)

plt.show()
