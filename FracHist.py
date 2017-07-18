import EvenMinusOdd as emo
import numpy as np

UV = emo.EvenMinusOdd()
UV.read_even_odd(False,True,'/Users/mike_e_dubs/python_stuff/uvfits/1061313128.uvfits')

H = UV.waterfall_hist_prepare((7000,19000))

#How many measurements are made per t-f??

N = UV.even.Nbls*UV.even.Npols
N = float(N)

#Express fraction instead of counts

H = H/N

H = np.sum(H, axis = 2)


UV.waterfall_hist_plot(H,'Fractional RFI Waterfall Histogram OBSID 1061313128, No Time Edge')
