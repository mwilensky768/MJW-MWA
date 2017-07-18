import numpy as np
import EvenMinusOdd as emo
from matplotlib import cm
from matplotlib import GridSpec

#To be run on MIT

def waterfall_settings(fig,ax,W,title,MAX):
    cax = ax.imshow(W,cmap = cm.coolwarm, interpolation = 'none')
    ax.set_title(title)
    ax.set_aspect(0.067)
    cbar = ax.colorbar(cax,ticks = [0.2*MAX*k for k in range(6)], use_gridspec = True)
    cbar.set_ticklabels([str(MAX*k*0.2) for k in range(6)])

Obsi = 1061311664 #First Obs in Golden Set
Obsf = 1061323008 #Last Obs in Golden Set

OBSLIST = [Obsi]
Step = [120,120,128,120]
k = 0

while OBSLIST[k] < Obsf:
    OBSLIST.append(OBSLIST[k] + Step[k%4])
    k += 1

Nobs = len(OBSLIST)


for l in range(Nobs):
    
    UV = emo.EvenMinusOdd()
    UV.read_even_odd(False,False,'/nfs/eor-11/r1/EoRuvfits/jd2456528v4_1/'+str(OBSLIST[l])+'/'+str(OBSLIST[l])+'.uvfits')
    AmpHist = UV.one_d_hist_prepare()
    AmpHistMax = max(AmpHist[0])
    WHist = UV.waterfall_hist_prepare((2000,21000))
    WMaxes = [np.amax(WHist[:,:,k]) for m in range(4)]

    fig = plt.figure()
    gs = GridSpec(3,2)

    axes = [plt.subplot(gs[1,0]),plt.subplot(gs[1,1]),plt.subplot(gs[2,0]),plt.subplot(gs[2,1]),plt.subplot(gs[0,:])]
    titles = ['XX','YY','XY','YX']
    
    for m in range(5):
        if m < 4:
            waterfall_settings(fig,axes[m],WHist[:,:,m],titles[m],WMaxes[m])

            if m == 0 or m == 2:
                axes[m].set_yticks([64*k for k in range(7)])
                axes[m].set_ylabel('Frequency Channel #')

            if m == 2 or m == 3:
                axes[m].set_xticks([7*k for k in range(5)])
                axes[m].set_xlabel('Time Pair')

        else:
            axes[m].hist(AmpHist, bins = 1000, range = (0,AmpHistMax), histtype = 'step', label = ('All','And','Neither','XOR'))
            axes[m].set_title('RFI Diagnostic Catalog '+str(OBSLIST[l]))
            axes[m].set_yscale('log',nonposy = 'clip')
            axes[m].set_xlabel('Amplitude ('+UV.even.vis_units+')')
            axes[m].set_ylabel('Counts')
            axes[m].set_xticks([0.1*AmpHistMax*k for k in range(11)])
            axes[m].ticklabel_format(style = 'sci', scilimits = (-1,1), axis = 'x')
            axes[m].legend()

    plt.savefig('/nfs/eor-00/h1/mwilensk/RFI_Diagnostic/'+str(OBSLIST[l])+'_RFI_Diagnostic.png')
    

    
