import matplotlib.pyplot as plt
import numpy as np

inpath = '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Long_Run_8s_Autos_Hists/'
Obs_text

with open('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Long_Run_8s_Autos_OBSIDS.txt') as f:
    obslist = f.read().split("\n")
    obslist.remove('')

n = np.load(inpath + obslist[0] + '_hist.npy')


for obs in obslist[1:]:
    n += np.load(inpath + obs + '_hist.npy')

bins = np.logspace(-3, 5, num=1001)
widths = np.diff(bins)
N = len(bins)

fig, ax = plt.subplots()
ax.step(bins[0:N - 1], n, where='pre')
ax.set_title('Long Run Unflagged Visibility Time-Difference Histogram, Autos/8s Removed')
ax.set_xlabel('Amplitude (UNCALIB)')
ax.set_ylabel('Counts')
ax.set_xscale('log', nonposy='clip')
ax.set_yscale('log', nonposy='clip')
ax.legend()

fig.savefig('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/Long_Run_8s_Autos_Hist.png')
