import rfipy as rfi
import matplotlib.pyplot as plt
import numpy as np

RFI = rfi.RFI(1061313008, '/Users/mike_e_dubs/python_stuff/uvfits/1061313008.uvfits')

T = RFI.ant_pol_prepare(38, 143)

fig, ax = plt.subplots()

RFI.waterfall_hist_plot(fig, ax, T, 'ant-pol test, t = 38, f = 143', np.amax(T), aspect_ratio=1,
                        fraction=False, y_type='ant-pol', x_type='ant-pol')

fig.savefig('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Ant_Pol_Plots/ant_pol_test_t38_f143_imag.png')
