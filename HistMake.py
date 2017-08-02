import pyuvdata as pyuv
import EvenMinusOdd as EMO
import numpy as np
import matplotlib.pyplot as plt

emo = EMO.EvenMinusOdd(False, False)

emo.read_even_odd('/Users/mike_e_dubs/python_stuff/uvfits/1061313008.uvfits')

W = emo.waterfall_hist_prepare((2000, 19000))
emo.waterfall_hist_plot()
