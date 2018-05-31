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
        cutoffs.append(rfiutil.emp_pdf(N, size=int(10**(m))))

    np.save('%srayleigh_lookup_%i.npy' % (outpath, m))
    pl.line_plot(fig, ax, [np.array(cutoffs), ], title='Cutoff vs. N, 1e%i' % (m),
                 xlabel='N', ylabel='Cutoff (median)', legend=False)
    fig.savefig('%scutoff_%i.png' % (outpath, m))
