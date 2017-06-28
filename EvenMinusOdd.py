import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pyuvdata as pyuv

class EvenMinusOdd:

    BEF = False #Remove the Coarse Band Edges?
    TEF = False #Remove the Time Edges?
    OBSID = 1061313008 #Which OBSID - be sure to set these three in the beginning
    even = pyuv.UVData()
    odd = pyuv.UVData()
    EMO = pyuv.UVData()
    
    def read_even_odd(self, filepath):
    
        self.even.read_uvfits(filepath)
        self.even.select(times = [UV.time_array[(2*k+1)*self.even.Nbls] for k in range(28)])

        self.odd.read_uvfits(filepath)
        self.odd.select(times = [UV.time_array[2*k*self.odd.Nbls] for k in range(28)])

        if self.BEF:

            LEdges = [16*p for p in range(24)]
            REdges = [15+16*p for p in range(24)]

            self.even.select(freq_chans) = [n for n in range(self.even.Nfreqs) if n not in LEdges and n not in REdges]
            self.odd.select(freq_chans) = [n for n in range(self.even.Nfreqs) if n not in LEdges and n not in REdges]

        if self.TEF:

            self.even.select(times = [UVeven.time_array[k*self.even.Nbls] for k in range(27)])
            self.odd.select(times = [UVodd.time_array[k*self.odd.Nbls] for k in range(1,28)])


        self.EMO.data_array = self.even.data_array-self.odd.data_array

    def 1D_hist_prepare(self, comp = 'amplitude', s = (0,4)): #s is a slice, see below, #comp may be amp or phase

        N = np.prod(self.EMO.data_array.shape)
        EMOV = np.reshape(self.EMO.data_array, N)
        evenFV = np.reshape(self.even.flag_array, N)
        oddFV = np.reshape(self.odd.flag_array, N)

        if comp is 'amplitude': #Find the amplitude of the difference
            EMOV = np.absolute(EMOV)
        elif comp is 'phase': #Find the phase of the difference
            EMOV = np.angle(EMOV)

        EMOVand = [EMOV[k] for k in range(N) if evenFV[k] and oddFV[k]]
        EMOVneither = [EMOV[k] for k in range(N) if not evenFV[k] and not oddFV[k]]
        EMOVXOR = [EMOV[k] for k in range(N) if evenFV[k]^oddFV[k]]

        Hdata = (EMOV, EMOVand, EMOVneither, EMOVXOR)[s[0]:s[1]] #Pick a slice for the histogram
        
        return(Hdata)

    def 1D_hist_plot(self, Hdata, Nbins, labels, title, comp = 'amplitude'):

        MAX = max(Hdata)
        if comp is 'amplitude':
            units = self.even.vis_units
        elif comp is 'phase':
            units = 'rad'

            
        plt.hist(Hdata, bins = Nbins, range = (0,MAX), histtype = 'step', label = labels)
        plt.title(title)
        plt.yscale('log', nonposy = clip)
        plt.xlabel(comp+' ('+units+')')
        plt.ylabel('Counts')
        plt.xticks([0.1*MAX*k for k in range(11)])
        plt.ticklabel_format(axis = 'x', style = 'sci', scilimits = [-1,1])
        plt.legend()

        plt.show()

    def waterfall_hist_prepare(self, band, comp = 'amplitude'): #band is a tuple (min,max)

        
        if comp is 'amplitude':
            Hdata = np.absolute(self.EMO.data_array)
        elif comp is 'phase':
            Hdata = np.angle(self.EMO.data_array)

        H = np.zeros([self.even.Nfreqs,self.even.Ntimes])

        for p in range(Hdata.shape[0]):#Nblts
            for q in range(Hdata.shape[2]):#Nfreqs
                for r in range(Hdata.shape[3]):#Npols
                    if min(band) < Hdata[p,0,q,r] < max(band):
                        H[q,p/self.even.Nbls] +=1

        return(H)

    def waterfall_hist_plot(self, H, title):


        MAX = np.amax(H)
        AVG = np.mean(H)
        
        fig,ax = plt.subplots()
        cax = ax.imshow(H, cmap = cm.coolwarm, interpolation = 'none')
        ax.set_title(title)
        ax.set_xticks([2*k for k in range(15)])
        ax.set_yticks([20*k for k in range(20)])
        ax.set_xlabel('Time (4s)')
        ax.set_ylabel('Frequency (channel #)')        
        ax.set_aspect(0.067)
        cbar = fig.colorbar(cax, ticks = [MAX*0.1*k for k in range(11)])
        cbar.set_ticklabels([str(MAX*0.1*k) for k in range(11)])
        

        plt.show()
        
