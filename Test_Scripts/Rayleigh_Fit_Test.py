import rfipy as rfi
import matplotlib.pyplot as plt
import numpy as np

RFI = rfi.RFI(1065538888, '/Users/mike_e_dubs/python_stuff/uvfits/1065538888.uvfits')

data, fit_params = RFI.one_d_hist_prepare()

fig, ax = plt.subplots()

RFI.one_d_hist_plot(fig, ax, data, 'Unflagged', '1065538888 Rayleigh Fit Test',
                    fit_params=fit_params)

fig.savefig('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/1065538888_Rayleigh_Fit_Test.png')

plt.show()
