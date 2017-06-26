import UVToys as UVT
import numpy as np

UVT.TimeFreqHist('/nfs/mwa-14/r1/EoRuvfits/jd2456528v4_1/1061313008.uvfits',1061313008, False, False)

bblow = [,]

bbhigh = [,]

bw = bbhigh - bblow + [1,1]

t = [,]

NRFI = [np.sum(G[[bblow[k]:bbhigh[k]],t[k]]) for k in range(2)]

Ntotal = [float(8128*4*x for x in bw)]

frac = [NRFI[k]/Ntotal[k] for k in range(2)]
