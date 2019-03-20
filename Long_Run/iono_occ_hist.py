import pickle
import numpy as np
from SSINS import plot_lib
import matplotlib.pyplot as plt
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('shape')
args = parser.parse_args()

indir = '/Users/mike_e_dubs/MWA/INS/Long_Run/Original_Jackknife'

with open('%s/long_run_original_occ_dict.pik' % indir, 'rb') as occ_file:
    occ_dict = pickle.load(occ_file)


x = []
y = []

edges = [16 * k for k in range(24)] + [15 + 16 * k for k in range(24)]
good_freqs = np.ones((384), dtype=bool)
good_freqs[edges] = 0

with open('%s/LR_iono.csv' % indir) as iono_file:
    csv_reader = csv.reader(iono_file, delimiter=',')
    for i, row in enumerate(csv_reader):
        if i:
            obs = row[0]
            if obs in occ_dict[5][args.shape]:
                x.append(float(row[-3]))
                if args.shape == 'total':
                    y.append(np.mean(occ_dict[5][args.shape][obs][good_freqs]))
                else:
                    y.append(occ_dict[5][args.shape][obs])

x_bins = np.linspace(1, 10, num=21)
y_bins = np.linspace(0, 0.5, num=21)
print(x_bins[16])

H, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])

H = np.flipud(H)

fig, ax = plt.subplots(figsize=(16, 9))
cax = ax.imshow(H)
cbar = fig.colorbar(cax, ax=ax)
ax.set_xlabel('RFI Occupancy Fraction')
ax.set_ylabel('Ionospheric Quality Metric')
cbar.set_label('Counts')
xticklabels = ['%.2f' % y_bins[tick] for tick in ax.get_xticks()[1:].astype(int)]
yticklabels = ['%.3f' % (x_bins[tick]) for tick in ax.get_yticks()[1:].astype(int)]
xticklabels.insert(0, '0')
yticklabels.reverse()
ax.set_xticklabels(xticklabels)
ax.set_yticklabels(yticklabels)
ax.set_title('%s Occupancy vs. Ionospheric Metric 2d Histogram' % args.shape)
fig.savefig('%s/iono_occ_%s_2dhist.png' % (indir, args.shape))
