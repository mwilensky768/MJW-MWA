import EvenMinusOdd as emo

UV = emo.EvenMinusOdd()

UV.read_even_odd(False,True,'/Users/mike_e_dubs/python_stuff/uvfits/1061313128.uvfits')

UV.one_d_hist_plot(UV.one_d_hist_prepare(), 1000, ('All','And','Neither','XOR'), '1061313128 EMO UV Amp. No Time Edge')


