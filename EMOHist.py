import pyuvdata as pyuv
import numpy as np
import matplotlib.pyplot as plt
import UVToys as UVT

def EMOHist(UV,OBSID,BEFilter,TEFilter):#(UVData(),int,bool,bool)   

    HData = UVT.EMO(UV, BEFilter, TEFilter)

    #construct the histograms
    plt.hist(HData, bins = 1000, range = (0,max(HData[0])), histtype = 'step', label = ('ALL','AND','NEITHER','XOR'))
    plt.title('EMO Visibility Amplitude ObsID '+str(OBSID)+'[no band/time edge]')
    plt.yscale('log', nonposy = 'clip')
    plt.xlabel('|Veven-Vodd| (uncalib)')
    plt.ylabel('Counts')
    plt.xticks([10000*k for k in range(9)])
    plt.legend()

    plt.show()


