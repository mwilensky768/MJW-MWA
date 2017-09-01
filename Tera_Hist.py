import matplotlib.pyplot as plt
import numpy as np

with open('/nfs/eor-00/h1/mwilensk/FHD/obs_list/beardsley_thesis_list.txt') as f:
    obslist = f.read().split("\n")
    obslist.remove('')

n1 = np.load('/nfs/eor-00/h1/mwilensk/long_run_hist_8s/' + str(obslist[0]) + '_hist.npy')
n2 = np.load('/nfs/eor-00/h1/mwilensk/long_run_hist/' + str(obslist[0]) + '_hist.npy')

for obs in obslist[1:]:
    n1 += np.load('/nfs/eor-00/h1/mwilensk/long_run_hist_8s/' + obs + '_hist.npy')
    n2 += np.load('/nfs/eor-00/h1/mwilensk/long_run_hist/' + obs + '_hist.npy')

bins = np.logspace(-3, 5, num=1001)
widths = np.diff(bins)
N = len(bins)

fig, ax = plt.subplots()
ax.step(bins[0:N - 1], n1, where='pre', label='8s removed')
ax.step(bins[0:N - 1], n2, where='pre', label='4s removed')
ax.set_title('Long Run Unflagged Visibility Time-Difference Histogram')
ax.set_xlabel('Amplitude (UNCALIB)')
ax.set_ylabel('Counts')
ax.set_xscale('log', nonposy='clip')
ax.set_yscale('log', nonposy='clip')
ax.legend()

fig.savefig('/nfs/eor-00/h1/mwilensk/long_run_hist_8s/long_run_unflagged_hist_together.png')
