import numpy as np
import matplotlib.pyplot as plt
import pyuvdata as pyuv
from matplotlib import cm
from math import floor, log10
from matplotlib.gridspec import GridSpec

class EvenMinusOdd:

    UV = pyuv.UVData()
    even = pyuv.UVData()
    odd = pyuv.UVData()
    EMO = pyuv.UVData()

    def __init__(self, BEF, TEF):

        self.BEF = BEF
        self.TEF = TEF
    
    def read_even_odd(self, filepath):
    
        self.UV.read_uvfits(filepath)

        if self.BEF:
            LEdges = [16*p for p in range(24)]
            REdges = [15+16*p for p in range(24)]

            self.UV.select(freq_chans = [n for n in range(self.dummy.Nfreqs) if n not in LEdges and n not in REdges])

        if self.TEF:

            self.UV.select(times = [self.UV.time_array[k*self.UV.Nbls] for k in range(1,55)])
            
        self.even = self.UV.select(times = [self.dummy.time_array[(2*k+1)*self.UV.Nbls] for k in range(28-self.TEF)], inplace = False)
        self.odd = self.UV.select(times = [self.dummy.time_array[2*k*self.UV.Nbls] for k in range(28-self.TEF)], inplace = False)

        
        self.EMO.data_array = self.even.data_array-self.odd.data_array

    def one_d_hist_prepare(self, comp = 'amplitude', s = (0,4)): 

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

        H = (EMOV, EMOVand, EMOVneither, EMOVXOR)[s[0]:s[1]] #Pick a slice for the histogram

        return(H)

    def one_d_hist_plot(self, H, Nbins, labels, title, comp = 'amplitude'): #Data/title are tuples if multiple hists

        MAXlist = [max(H[k]) for k in range(len(H))]
        MAX = max(MAXlist)
        if comp is 'amplitude':
            units = self.even.vis_units
        elif comp is 'phase':
            units = 'rad'

            
        plt.hist(H, bins = Nbins, range = (0,MAX), histtype = 'step', label = labels)
        plt.title(title)
        plt.yscale('log', nonposy = 'clip')
        plt.xlabel(comp+' ('+units+')')
        plt.ylabel('Counts')
        plt.xticks([0.1*MAX*k for k in range(11)])
        plt.ticklabel_format(axis = 'x', style = 'sci', scilimits = [-1,1])
        plt.legend()

        plt.show()

    def waterfall_hist_prepare(self, band, comp = 'amplitude', fraction = True, flag_slice = 'Neither'): #band is a tuple (min,max)

        
        if comp is 'amplitude':
            data = np.absolute(self.EMO.data_array)
        elif comp is 'phase':
            data = np.angle(self.EMO.data_array)


        H = np.zeros([self.even.Ntimes,self.even.Nfreqs,self.even.Npols])

        if flag_slice is 'Neither':
            for p in range(data.shape[0]):
                for q in range(data.shape[2]):
                    for r in range(data.shape[3]):
                        if min(band) < data[p,0,q,r] < max(band) and not self.even.flag_array[p,0,q,r] and not self.odd.flag_array[p,0,q,r]:
                            H[p/self.UV.Nbls,q,r] += 1
        else:
            for p in range(data.shape[0]):#Nblts/2
                for q in range(data.shape[2]):#Nfreqs
                    for r in range(data.shape[3]):#Npols
                        if min(band) < data[p,0,q,r] < max(band):
                            H[p/self.UV.Nbls,q,r] += 1

        if fraction is True:
            N = float(self.UV.Nbls*self.UV.Npols)
            H = H/N
        
        return(H)


    def waterfall_hist_plot(self, H, title):


        MAX = np.amax(H)
        AVG = np.mean(H)
        
        fig,ax = plt.subplots()
        cax = ax.imshow(H,cmap = cm.coolwarm, interpolation = 'none')
        ax.set_title(title)
        ax.set_xticks([16*k for k in range(25)])
        ax.set_yticks([2*k for k in range(15)])
        ax.set_xlabel('Time Pair')
        ax.set_ylabel('Frequency (channel #)')        
        ax.set_aspect(15)
        cbar = fig.colorbar(cax, ticks = [MAX*0.1*k for k in range(11)])
        cbar.set_ticklabels([str(MAX*0.1*k) for k in range(11)])
        

        plt.show()

    def rfi_catalog(self, obslist, main_title, thresh_min = 2000, flag_slice = 'Neither'): #obslist should be a list of integers (OBSID's)

        Nobs = len(obslist)
        
        for m in range(Nobs):
            self.read_even_odd('/nfs/eor-11/r1/EoRuvfits/jd2456528v4_1/'+str(obslist[m])+'/'+str(obslist[m])+'.uvfits')

            H = self.one_d_hist_prepare()
            MAXH = max(H[0])

            W = self.waterfall_hist_prepare((thresh_min, MAXH), flag_slice = flag_slice)
            MAXWlist = [np.amax(W[k]) for k in range(len(W))]

            
            fig = plt.figure(figsize = (14,8))
            gs = GridSpec(3,2)
            pol_titles = ['XX','YY','XY','YX']
            axes = [plt.subplot(gs[1,0]),plt.subplot(gs[1,1]), plt.subplot(gs[2,0]),plt.subplot(gs[2,1]),plt.subplot(gs[0,:])]
            plt.subplots_adjust(left = 0.13, bottom = 0.11, right = 0.90, top = 0.88, wspace = 0.20, hspace = 0.46)
            colormax = [max(MAXWlist[0:2]), max(MAXWlist[2:4])]

            def colormax(p):
                if p in [0,1]:
                    return(max(MAXWlist[0:2]))
                else:
                    return(max(MAXWlist[2:4]))
                    

            def sigfig(x,s = 4): #s is number of sig-figs
                if x == 0:
                    return(0)
                else:
                    n = int(floor(log10(abs(x))))
                    y = 10**n*round(10**(-n)*x, s-1)
                    return(y)

            def waterfall_settings(fig, axis, pol_title, W, p):
                
                axis.set_title(pol_title)
                cax = axis.imshow(W, cmap = cm.binary, interpolation = 'none', vmin = 0, vmax = colormax(p))
                axis.set_aspect(6)
                y_ticks = [7*k for k in range(5)]
                x_ticks = [64*k for k in range(7)]
                color_ticks = [0.2*vmax*k for k in range(6)]
                axis.set_yticks(y_ticks)
                axis.set_xticks(x_ticks)
                cbar = fig.colorbar(cax, ax = axis, ticks = color_ticks)
                cbar.set_ticklabels([str(sigfig(color_ticks[k])) for k in range(6)])

            for n in range(5):
                if n < 4:
                    waterfall_settings(fig,axes[n],pol_titles[n]+' '+flag_slice,W[:,:,n],n) #Common Waterfall Settings
                    if n in [0,2]: #Some get axis labels others do not
                        axes[n].set_ylabel('Time Pair')
                    if n in [2,3]:
                        axes[n].set_xlabel('Frequency (Mhz)')
                        x_ticks_labels = [str(sigfig(self.UV.freq_array[0,64*k]*10**(-6))) for k in range(6)]
                        x_ticks_labels.append(str(sigfig((self.UV.freq_array[0,-1]+self.UV.channel_width)*10**(-6))))
                        axes[n].set_xtickslabels(x_ticks_labels)
                    if n in [0,1]:
                        axes[n].set_xtickslabels([])
                    if n in [1,3]:
                        axes[n].set_yticklabels([])
                else:
                    axes[n].hist((H[2],H[0],H[1],H[3]),bins = 1000, range = (0,MAXH), histtype = 'step', label = ('Neither','All','And','XOR'))
                    axes[n].set_title('RFI Catalog '+str(obslist[m]))
                    axes[n].set_yscale('log',nonposy = 'clip')
                    axes[n].set_xscale('log',nonposy='clip')
                    axes[n].set_xlabel('Amplitude (UNCALIB)')
                    axes[n].set_ylabel('Counts')
                    axes[n].set_xticks([10**(k-10)*MAXH for k in range(11)])
                    axes[n].ticklabel_format(style = 'sci', scilimits = (-1,1), axis = 'x')
                    axes[n].legend()

            plt.savefig('/nfs/eor-00/h1/mwilensk/RFI_Diagnostic/'+str(obslist[m])+'_RFI_Diagnostic.png')
