from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import scipy.stats

parser = argparse.ArgumentParser()
parser.add_argument('outdir', help='The directory to save outputs to')
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

lam = 50.1
data = np.random.poisson(lam=lam, size=int(1e5))
counts, bins = np.histogram(data, bins='auto')
widths = np.diff(bins)
centers = bins[:-1] + 0.5 * widths
pdf = counts / (widths * 1e5)
print('The pdf integrates to %f' % np.sum(pdf * widths))

fig, ax = plt.subplots(figsize=(14, 8), nrows=2)
ax[0].plot(centers, counts, drawstyle='steps-mid')
ax[0].set_ylabel('Counts')
ax[0].set_title('Simulated Poisson Histogram')
ax[1].plot(centers, pdf)
ax[1].set_ylabel('Density')
ax[1].set_title('Empirical Poisson PDF')
for i in range(2):
    ax[i].set_xlabel('Events')
    ax[i].set_yscale('log', nonposy='clip')

fig.savefig('%s/Poisson_Semilog.png' % args.outdir)
plt.close(fig)

BG_prob = scipy.stats.poisson.sf(80, lam)
sig_prob = scipy.stats.poisson.cdf(80, lam)
sink_prob = scipy.stats.poisson.sf(30, lam)
sig_prob_sig = (80 - lam) / np.sqrt(lam)
sink_prob_sig = (30 - lam) / np.sqrt(lam)

print('The probability that the background gave me at least 80 events in the interval is %f' % BG_prob)
print('The probability that this is a signal is %f' % sig_prob)
print('Therefore the probability of my detection given the data is %f' % sig_prob)
print('The probability that the background gave me at least 30 counts is %f' % sink_prob)
print('This is the same as the probability that I found the sink.')
print('The sigma of the first detection is %f' % sig_prob_sig)
print('The sigma of the sink detection is %f' % sink_prob_sig)

for sig_b in [1, 10]:
    A = np.random.normal(size=int(1e6))
    B = np.random.normal(size=int(1e6), scale=sig_b)

    mag = np.sqrt(A**2 + B**2)
    mag_counts, mag_bins = np.histogram(mag, bins='auto')
    mag_widths = np.diff(mag_bins)
    mag_centers = mag_bins[:-1] + 0.5 * mag_widths
    mag_pdf = mag_counts / (1e6 * mag_widths)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(mag_centers, mag_pdf)
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Density')
    ax.set_yscale('log', nonposy='clip')
    fig.savefig('%s/Double_Gauss_Amp_%i.png' % (args.outdir, sig_b))
    plt.close(fig)

bins_x = np.arange(-50, 51)
counts_2d, bins_x, bins_y = np.histogram2d(A, B, bins=bins_x)
fig, ax = plt.subplots(figsize=(14, 8))
cax = ax.imshow(counts_2d.T)
cbar = fig.colorbar(cax, ax=ax)
ax.set_xlabel('Length of A')
ax.set_ylabel('Length of B')
cbar.set_label('Counts')
xticklabels = ['%.0f' % bins_x[tick] for tick in ax.get_xticks()[1:].astype(int)]
yticklabels = ['%.0f' % bins_y[tick] for tick in ax.get_yticks()[1:].astype(int)]
xticklabels.insert(0, '0')
yticklabels.insert(0, '0')
ax.set_xticklabels(xticklabels)
ax.set_yticklabels(yticklabels)
fig.savefig('%s/hist2d.png' % args.outdir)
