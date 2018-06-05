import rfiutil
import matplotlib.pyplot as plt
import plot_lib as pl
import os

outpath = '/lookup_table/'
if not os.path.exists(outpath):
    os.makedirs(outpath)

for m in range(6, 10):
    fig, ax = plt.subplots(figsize=(14, 8))
    cutoffs = []
    for N in range(768 * 56):
        model, bins, cutoff, var, mean = rfiutil.emp_pdf(N, size=int(10**(m)))
        cutoffs.append(cutoff)
        fig_hist, ax_hist = plt.subplots(figsize=(14, 8))
        pl.one_d_hist_plot(fig_hist, ax_hist, bins, [model, ], xlog=False,
                           xlabel='Amplitude (Median)', ylabel='Counts',
                           title='N = %i k = %i Simulated Hist' % (N, m))
        fig.savefig('%ssim_hist_N%i_m%i.png' % (outpath, N, m))
        plt.close(fig_hist)

    np.save('%srayleigh_lookup_%i.npy' % (outpath, m))
    pl.line_plot(fig, ax, [np.array(cutoffs), ], title='Cutoff vs. N, 1e%i' % (m),
                 xlabel='N', ylabel='Cutoff (median)', legend=False)
    fig.savefig('%scutoff_%i.png' % (outpath, m))
    plt.close(fig)
