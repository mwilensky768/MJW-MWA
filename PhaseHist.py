import EvenMinusOdd as emo

IB = emo.EvenMinusOdd()

IB.read_even_odd(False,False,'/Users/mike_e_dubs/python_stuff/uvfits/1061313008.uvfits')

G = IB.one_d_hist_prepare(comp = 'phase')

IB.one_d_hist_plot(G,100,('All','And','Neither','XOR'),'EMO Phase 1061313008',comp = 'phase')
