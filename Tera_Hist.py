import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

inpath = '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Long_Run_8s_Autos_Hists/'
fit = True

with open('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Long_Run_8s_Autos_OBSIDS.txt') as f:
    obslist = f.read().split("\n")
    obslist.remove('')

n = np.load(inpath + obslist[0] + '_hist.npy')


for obs in obslist[1:]:
    n += np.load(inpath + obs + '_hist.npy')

bins = np.logspace(-3, 5, num=1001)
widths = np.diff(bins)
centers = bins[:-1] + 0.5 * widths
N = len(bins)

if fit:
    def rayleigh(centers, area, sigma):
        return((area / sigma**2) * centers * np.exp(-centers**2 / (2 * sigma**2)))
    cond = np.logical_and(0 < centers, centers < 1000)
    sigma = centers[n == np.amax(n)][0]
    area = np.sum(n * widths)
    popt, pcov = curve_fit(rayleigh, centers, n, p0=[area, sigma])
    curve = rayleigh(centers[cond], popt[0], popt[1])

fig, ax = plt.subplots(figsize=(14, 8))
ax.step(bins[np.append(cond, False)], n[cond], where='pre', label='Histogram')
ax.plot(centers[np.logical_and(0 < centers, centers < 1000)], curve, label='Fit: sigma = ' + str(sigma))
ax.set_title('Long Run Unflagged Visibility Time-Difference Histogram, Autos/8s Removed')
ax.set_xlabel('Amplitude (UNCALIB)')
ax.set_ylabel('Counts')
ax.set_xscale('linear')
ax.set_yscale('linear')
ax.legend()

fig.savefig('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Long_Run_8s_Autos_Hist.png')
