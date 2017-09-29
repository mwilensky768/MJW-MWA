import numpy as np
import UVToys as UVT
import SumThreshold as ST
import matplotlib.pyplot as plt
from matplotlib import cm
import SIR

#This script will attempt to flag 1061313128D1 with sumthreshold

UV = np.load('/Users/radcosgroup/UVFITS/1061313128D1.npy')
UVamp = abs(UV)

#Array should be constant time-slices (flagging frequency sequences)

M = 20 #Will flag subsequences of length M

SubS = np.zeros([56,384-M+1], dtype = float)

for m in range(0,56): #Generate array of subsequence sums
    SubS[m,:] = UVT.SubSequenceSum(UVamp[m,:], M)

Stats = UVT.UVstats(SubS) #return some stats

chi = (Stats[0] + 5*np.sqrt(Stats[1]))/M #Set threshold to mean plus 5 sigma
eta = 0.05 #set SIR aggression

Flags = np.zeros([56,384], dtype = bool)

for k in range(0,2):
    for m in range(0,56): #Apply SumThreshold
        Flags[m,:] = ST.SumThreshold(UVamp[m,:], Flags[m,:], M, chi)

    for m in range(0,56): #Apply SIR
        Flags[m,:] = SIR.SIROperator(Flags[m,:], eta)
        

#The following code plots the flag mask

fig, ax = plt.subplots()

cax = ax.imshow(Flags, cmap = cm.binary)
ax.set_title('1061313128 Baseline 1 XX Flag ST Test')

cbar = fig.colorbar(cax, ticks = [0,1])
cbar.ax.set_yticklabels(['Unflagged', 'Flagged'])

plt.show()
