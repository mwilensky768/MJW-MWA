import pyuvdata as pyuv
import numpy as np
import matplotlib.pyplot as plt

UV = pyuv.UVData()
UV.read_uvfits('/Users/mike_e_dubs/python_stuff/smaller_uvfits/s1061313008.uvfits')

baselines = []

for m in range(UV.Nants_telescope):
    baselines.append(UV.antnums_to_baseline(m, m))

print('There are ' + str(len(np.unique(baselines))) + ' baselines mapped to by the autocorrelations.')
print('We went through ' + str(m + 1) + ' antennas.')

print(baselines)
