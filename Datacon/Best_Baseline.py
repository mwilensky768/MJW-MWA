from pyuvdata import UVData
import numpy as np
from SSINS import plot_lib
from matplotlib import cm
import matplotlib.pyplot as plt
import os

inpath = '/Volumes/Faramir/uvfits/1066742016.uvfits'
outpath = '/Users/mikewilensky/General'


def dist(A, B):
    d = np.sum((A - B) ** 2)
    return(d)


UV = UVData()
UV.read(inpath, file_type='uvfits', polarizations=-5)
UV.select(ant_str='cross', times=np.unique(UV.time_array)[1:-3])

UV.data_array = np.reshape(UV.data_array,
                           (UV.Ntimes, UV.Nbls, UV.Nspws, UV.Nfreqs, UV.Npols))
UV.data_array = np.absolute(np.diff(UV.data_array, axis=0))
ins = UV.data_array.mean(axis=1)
i_min = -np.inf
d_min = np.inf
for i in range(UV.Nbls):
    d = dist(UV.data_array[:, i, 0, :, 0], ins[:, 0, :, 0])
    if d < d_min:
        i_min = i
        d_min = d
print(i_min)
