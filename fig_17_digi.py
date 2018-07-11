import numpy as np
import argparse
import itertools
import plot_lib as pl
import matplotlib.pyplot as plt
import os
import time


class digital_signal:

    def __init__(self, n, anal_sig, q=1):
        # n is the number of levels, q is the spacing of the quantizer
        self.n = n
        self.q = q

        self.heights, self.jumps, self.bins = self.LevelCalc()
        self.sig = self.heights[(np.digitize(anal_sig, self.bins) - 1)]
        self.wid = np.sqrt(np.mean(self.sig**2))

    def LevelCalc(self):
        if self.n % 2:
            heights = self.q * np.arange(-np.floor(self.n / 2), np.floor(self.n / 2) + 1)
        else:
            heights = self.q * np.arange(-(self.n - 1), self.n + 1, 2)
        jumps = np.diff(heights)
        thresh = heights[:self.n - 1] + 0.5 * jumps
        bins = np.insert(thresh, 0, -np.inf)
        bins = np.append(bins, np.inf)

        return(heights, jumps, bins)


if __name__ == "__main__":
    print('Started at %s' % (time.strftime('%H:%M:%S')))

    parser = argparse.ArgumentParser()
    parser.add_argument("n", action='store', nargs=1, type=int, help="The number of levels")
    parser.add_argument("q", action='store', nargs=1, type=int, help="The spacing between the levels")
    parser.add_argument("s", action='store', nargs=1, type=int, help="The exponent of the signal length")
    parser.add_argument("outpath", action='store', nargs=1, help="Destination for figure")
    args = parser.parse_args()

    sigma = np.arange(0.5, 3.25, 0.25)
    rhos = np.array([0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999])
    SNR = np.sqrt(rhos / (1 - rhos))
    A = SNR / np.sqrt(1 + SNR**2)
    B = 1 / np.sqrt(1 + SNR**2)

    fig, ax = plt.subplots(nrows=3, sharex=True)
    fig.suptitle('Figure 17 Wilensky Reproduction (Simulation)')
    title = ''
    xlabel = r'$\rho$'
    ylabels = ['kappa', 'rho_hat', 'frac_diff']
    legend = False
    plot_kwargs = {'title': title,
                   'xlabel': xlabel,
                   'legend': legend}

    for combo in itertools.combinations_with_replacement(sigma, 2):
        corr = np.zeros(len(rhos))
        rho_quant = np.copy(corr)
        frac = np.copy(corr)

        for i, (a, b) in enumerate(zip(A, B)):
            # cov = np.array([[combo[0]**2, rho * combo[0] * combo[1]],
                            # [rho * combo[0] * combo[1], combo[1]**2]])

            # anal_sigs = np.random.multivariate_normal([0, 0], cov, size=10**args.s[0])
            s = np.random.normal(size=10**args.s[0])
            n1 = np.random.normal(size=10**args.s[0])
            n2 = np.random.normal(size=10**args.s[0])
            anal_sig_1 = combo[0] * (a * s + b * n1)
            anal_sig_2 = combo[1] * (a * s + b * n2)

            dig_sig_1 = digital_signal(args.n[0], anal_sig_1, args.q[0])
            dig_sig_2 = digital_signal(args.n[0], anal_sig_2, args.q[0])

            corr[i] = np.mean(dig_sig_1.sig * dig_sig_2.sig)
            if not i:
                k = rhos[i] / corr[i]
            rho_quant[i] = k * corr[i]
            frac[i] = rho_quant[i] / rhos[i] - 1
            data = [corr, rho_quant, frac]

        for axis, y, ylabel in zip(ax, data, ylabels):
            plot_args = (fig, axis, rhos, [y, ])
            plot_kwargs.update({'ylabel': ylabel})
            pl.line_plot(*plot_args, **plot_kwargs)

    if not os.path.exists(args.outpath[0]):
        os.makedirs(args.outpath[0])
    fig.savefig('%s/digital_correlation_sim_n%i_q%i_s%i.png' % (args.outpath[0],
                                                                args.n[0],
                                                                args.q[0],
                                                                args.s[0]))
    print('Ended at %s' % (time.strftime('%H:%M:%S')))
