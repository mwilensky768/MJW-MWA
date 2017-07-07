import pyuvdata as pyuv
import numpy as np
import matplotlib.pyplot as plt

UV = pyuv.UVData()

UV.read_uvfits('/Users/mike_e_dubs/python_stuff/uvfits/1061313128.uvfits')

UV.select(blt_inds = [1+8128*m for m in range(56)])

for k in range(6):

    t = 18+k
    data = np.absolute(UV.data_array[t,0,:,0])
    MAX = np.amax(data)
    
    plt.figure(k+1)

    plt.plot(data)
    plt.xlabel('Frequency (channel #)')
    plt.ylabel('|V| (uncalib)')
    plt.title('1061313128 Baseline 1 XX t = '+str(t))
    plt.xticks([16*k for k in range(25)])
    plt.yticks([0.1*MAX*k for k in range(10)])

plt.show()
