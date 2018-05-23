import numpy as np
import matplotlib.pyplot as plt
import plot_lib as pl
import rfiutil

for N in np.arange(10, 110, 10):
    x, pdf = rfiutil.rayleigh_convolve(N, 10000)
    fig, ax = plt.subplots(figsize=(14, 8))
    pl.line_plot(fig, ax, [pdf, ], title='Rayleigh Convolve Test', xlabel='',
                 ylabel='', legend=False)
    fig.savefig('/Users/mike_e_dubs/MWA/Test_Plots/rayl_con/%i.png' % (N))
    plt.close(fig)
