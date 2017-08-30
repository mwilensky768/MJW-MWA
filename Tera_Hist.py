import matplotlib.pyplot as plt
import numpy as np

with open('/nfs/eor-00/h1/mwilensk/FHD/obs_list/beardsley_thesis_list.txt') as f:
    obslist = f.read().split("\n")

n = np.load('/nfs/eor-00/h1/mwilensk/long_run_hist/' + str(obslist[0]) + '_hist.npy')

for k in range(1, len(obslist)):
    n += np.load('/nfs/eor-00/h1/mwilensk/long_run_hist/' + str(obslist[k]) + '_hist.npy')

bins = np.logspace(-3, 5, num=1001)
widths = np.diff(bins)
N = len(bins)

fig, ax = plt.subplots()
ax.bar(bins[0, N - 1], n, width=widths, color='white', edgecolor='blue')
ax.set_title('Long Run Unflagged Visibility Time-Difference Histogram')
ax.set_xlabel('Amplitude (UNCALIB)')
ax.set_ylabel('Counts')
ax.set_xscale('log', nonposy='clip')
ax.set_yscale('log', nonposy='clip')
