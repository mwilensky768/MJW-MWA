import numpy as np
import matplotlib.pyplot as plt
import plot_lib as pl

frac = []
for k in range(41):
    frac.append(np.load('/Users/mike_e_dubs/MWA/Test_Plots/1061313128_actual_flag_no_time_recalc/arrs/Flag_Fraction_%i.npy' % (k)))

frac = np.array(frac)
hist, bins = np.histogram(frac, bins='auto')
print(max(frac))
print(min(frac))


fig, ax = plt.subplots(figsize=(14, 8))
pl.one_d_hist_plot(fig, ax, bins, [hist, ], xlog=False, ylog=False,
                   xlabel='Fraction Baseline Flagged',
                   title='1061313128 INS Flag Baseline Fraction Histogram',
                   legend=False)

fig.savefig('/Users/mike_e_dubs/MWA/Test_Plots/1061313128_actual_flag_no_time_recalc/figs/fraction_baselines_hist.png')
