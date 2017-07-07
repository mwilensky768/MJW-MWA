import EvenMinusOdd as emo
import matplotlib.pyplot as plt

UV = emo.EvenMinusOdd()
UV.read_even_odd('/Users/mike_e_dubs/python_stuff/uvfits/1061313008.uvfits')

H = UV.waterfall_hist_prepare((2000,19000))

#How many measurements are made per t-f??

N = UV.even.Nbls*UV.even.Npols
N = float(N)

#Express fraction instead of counts

H = H/N


UV.waterfall_hist_plot(H[:,:,0],'Fractional RFI Waterfall Histogram OBSID 1061313008 XX')


UV.waterfall_hist_plot(H[:,:,1],'Fractional RFI Waterfall Histogram OBSID 1061313008 YY')
