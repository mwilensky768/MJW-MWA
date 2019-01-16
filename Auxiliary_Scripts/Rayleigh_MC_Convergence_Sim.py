import numpy as np
from scipy.stats import norm, kstest
import argparse
from SSINS import plot_lib
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('Nmc', type=int)
parser.add_argument('size', type=int)
args = parser.parse_args()

bins = np.linspace(0, 5, num=101)
cdf = norm.cdf
for Navg in [2**i for i in range(11)]:
    dist_kwargs = {'loc': np.sqrt(np.pi / 2), 'scale': np.sqrt((2 - np.pi / 2) / Navg)}
    print('Using Navg=%i' % Navg)
    counts = np.zeros([args.Nmc, len(bins) - 1])
    D = np.zeros(args.Nmc)
    p = np.zeros(args.Nmc)
    for k in range(args.Nmc):
        avg_data = np.random.rayleigh(size=[args.size, Navg]).mean(axis=1)
        counts[k], _ = np.histogram(avg_data, bins=bins)
        D[k], p[k] = kstest(avg_data, 'norm', args=(np.sqrt(np.pi / 2), np.sqrt((2 - np.pi / 2) / Navg)))
    print(p.mean())
    counts_end = counts.mean(axis=0)
    counts_bars = np.sqrt(np.var(counts, axis=0))
    counts_end = np.append(counts_end, 0)
    counts_bars = np.append(counts_bars, 0)
    print('Done histogramming data')
    D_counts, D_bins = np.histogram(D, bins='auto')
    print('Done histogramming test statistics')
    p_bins = np.logspace(np.log10(p.min()) - 1, np.log10(p.max()) + 1, num=26)
    p_counts, p_bins = np.histogram(p, bins=p_bins)
    print('Done histogramming p-values')
    D_counts = np.append(D_counts, 0)
    p_counts = np.append(p_counts, 0)

    # pdf = counts_end / (args.size * np.diff(bins))
    # pdf_bars = counts_bars / (args.size * np.diff(bins))
    # pdf = np.append(pdf, 0)
    # pdf_bars = np.append(pdf_bars, 0)

    standard_prob = cdf(bins[1:], **dist_kwargs) - cdf(bins[:-1], **dist_kwargs)
    standard_counts = standard_prob * args.size
    standard_errors = np.sqrt(args.size * standard_prob * (1 - standard_prob))
    standard_counts = np.append(standard_counts, 0)
    standard_errors = np.append(standard_errors, 0)
    print('Done making gaussian histogram')
    # standard_pdf = np.append(standard_pdf, 0)

    fig, ax = plt.subplots(figsize=(16, 9))
    plot_lib.error_plot(fig, ax, bins, counts_end, label='Convolved Rayleigh')
    plot_lib.error_plot(fig, ax, bins, standard_counts, yerr=standard_errors, label='Gaussian', title='N_avg=%i' % Navg)
    fig.savefig('/Users/mike_e_dubs/MWA/Rayleigh_Convergence_Sim_%i.png' % Navg)
    plt.close(fig)
    print('Done plotting distributions')

    fig_test, ax_test = plt.subplots(figsize=(16, 9), nrows=2)
    plot_lib.error_plot(fig_test, ax_test[0], D_bins, D_counts, label='KS-Test Statistic', title='N_avg=%i' % Navg)
    plot_lib.error_plot(fig_test, ax_test[1], p_bins, p_counts, label='p-value', xscale='log')
    fig_test.savefig('/Users/mike_e_dubs/MWA/Rayleigh_Convergence_Sim_KS_%i.png' % Navg)
    print('Done plotting test statistics and p-values')
