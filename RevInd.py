import UVToys as UVT
import numpy as np

G = UVT.WaterfallHist('/nfs/mwa-14/r1/EoRuvfits/jd2456528v4_1/1061313008.uvfits', False, False)


N = 8128*4 #Number of measurements per bin = Nbsl*Npol (for THIS OBSID)

H = G/N

UVT.WaterfallHistPlot(H,1061313008,False,False)

