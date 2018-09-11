import numpy as np
import matplotlib.pyplot as plt
import os

outdir = '/Users/mike_e_dubs/MWA/Test_Plots/LLSE_Test'
if not os.path.exists(outdir):
    os.makedirs(outdir)
np.random.seed(seed=1)

for i in range(100):
    y = np.random.normal(size=51, loc=200, scale=1.16)
    x = np.arange(51)
    A = np.vstack((x, np.ones(len(x)))).T
    coeff = np.linalg.lstsq(A, y)[0]
    line = coeff[0] * x + coeff[1]
    print('data mean is %f' % np.mean(y))
    print('fit mean is %f' % np.mean(line))
    y_sub = y / line - 1

    fig, ax = plt.subplots(nrows=2)
    ax[0].scatter(x, y)
    ax[0].plot(x, line)
    ax[1].scatter(x, y_sub, c='r')

    fig.savefig('%s/fig_%i.png' % (outdir, i))
    plt.close(fig)
