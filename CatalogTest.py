import matplotlib.pyplot as plt
import EvenMinusOdd as emo
from matplotlib import cm
import numpy as np
from matplotlib.gridspec import GridSpec
from math import floor, log10

EMO = emo.EvenMinusOdd() #Make EMO instance

EMO.read_even_odd(False,False,'/Users/mike_e_dubs/python_stuff/uvfits/1061313008.uvfits') #Read data in

Q = EMO.one_d_hist_prepare() #Prepare 1d hist data
MAXQ = max(Q[0])

W = EMO.waterfall_hist_prepare((2000,19000)) #Prepare waterfall hist data
MAXW = [np.amax(W[:,:,r]) for r in range(4)] #Find some relevant metadata for later


fig = plt.figure(figsize = (14,8)) #Prepare figure and subplots
gs = GridSpec(3,2)
titles = ['XX','YY','XY','YX']
axes = [plt.subplot(gs[1,0]),plt.subplot(gs[1,1]), plt.subplot(gs[2,0]),plt.subplot(gs[2,1]),plt.subplot(gs[0,:])]
plt.subplots_adjust(left = 0.13, bottom = 0.11, right = 0.90, top = 0.88, wspace = 0.20, hspace = 0.46)

def maxw_selector(p):
    if p == 0 or p ==1:
        return(max(MAXW[0:2]))
    if p == 2 or p ==3:
        return(max(MAXW[2:4]))

def sigfig(x,r):#A useful function to make the plots prettier
    if x == 0:
        return(0)
    else:
        n = int(floor(log10(abs(x))))
        y = 10**n*round(10**(-n)*x, r-1)
        return(y)

def waterfall_settings(fig,axis,title,W,p): #A useful function to avoid lots of typing
    
    axis.set_title(title)
    cax = axis.imshow(W, cmap = cm.coolwarm, interpolation = 'none', vmin = 0, vmax = maxw_selector(p))
    axis.set_aspect(6)
    y_ticks = [7*k for k in range(5)]
    x_ticks = [64*k for k in range(7)]
    axis.set_yticks(y_ticks)
    axis.set_xticks(x_ticks)
    cbar = fig.colorbar(cax, ax = axis, ticks = [0.2*maxw_selector(p)*k for k in range(6)])
    cbar.set_ticklabels([str(sigfig(0.2*maxw_selector(p)*k, 3)) for k in range(6)])
    


for m in range(5): #Create the plots
    if m < 4:
        waterfall_settings(fig,axes[m],titles[m],W[:,:,m],m) #Common Waterfall features
        
        if m == 0 or m == 2: #Some get axis labels and others do not
            axes[m].set_ylabel('Time Pair')

        if m == 2 or m == 3: #XY and YX have freq ticks
            axes[m].set_xlabel('Frequency (MHz)')
            x_ticks_labels = [str(sigfig(EMO.even.freq_array[0,64*k]*10**(-6),4)) for k in range(6)]
            x_ticks_labels.append(str(sigfig((EMO.even.freq_array[0,383]+EMO.even.channel_width)*10**(-6),4)))
            axes[m].set_xticklabels(x_ticks_labels)
            

        if m == 0 or m == 1: #XX and YY share a color bar
            axes[m].set_xticklabels([])
                        
        if m == 1 or m == 3:
            axes[m].set_yticklabels([])
            

    else: #1-D Hist feature
        axes[m].hist(Q,bins = 1000, range = (0,MAXQ), histtype = 'step', label = ('All','And','Neither','XOR'))
        axes[m].set_title('RFI Catalog Test 1061313008')
        axes[m].set_yscale('log',nonposy = 'clip')
        axes[m].set_xlabel('Amplitude (UNCALIB)')
        axes[m].set_ylabel('Counts')
        axes[m].set_xticks([0.1*n*MAXQ for n in range(11)])
        axes[m].ticklabel_format(style = 'sci', scilimits = (-1,1), axis = 'x')
        axes[m].legend()

plt.show()
