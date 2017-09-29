import pyuvdata as pyuv
import numpy as np

UV1 = pyuv.UVData()
UV2 = pyuv.UVData()

UV1.read_uvfits('/Users/radcosgroup/UVFITS/1061313128.uvfits')
UV2.read_uvfits('/Users/radcosgroup/UVFITS/1061318864.uvfits')

QW1 = np.zeros([56,384], dtype = complex)
QW2 = np.copy(QW1) #It so happens that these Obs. ID's have similar metadata

FA1 = np.zeros([56,384], dtype = bool) #Dummy flag arrays
FA2 = np.copy(FA1)

#These loops just pick one baseline's time-frequency data for the Obs. ID
for p in range(0,UV1.Ntimes):
    for q in range(0,UV1.Nfreqs):
        QW1[p,q] = UV1.data_array[0+56*p,0,q,0]
        FA1[p,q] = UV1.flag_array[0+56*p,0,q,0]

for p in range(0,UV2.Ntimes):
    for q in range(0,UV2.Nfreqs):
        QW2[p,q] = UV2.data_array[0+56*p,0,q,0]
        FA2[p,q] = UV2.flag_array[0+56*p,0,q,0]



