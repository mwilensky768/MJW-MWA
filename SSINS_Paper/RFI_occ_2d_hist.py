import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.special import erfc
import argparse
from SSINS.util import make_obslist

parser = argparse.ArgumentParser()
parser.add_argument('--obslist')
args = parser.parse_args()

path = '/Users/mike_e_dubs/MWA/INS/Long_Run/Original_Jackknife_Revamp_Complete/occ_dict.pik'
outpath = '/Users/mike_e_dubs/Documents/My_Papers/SSINS_Paper'

with open(path, 'rb') as file:
    occ_dict = pickle.load(file)

subdict = occ_dict[5]['total']

x_data = np.zeros([len(subdict.keys()), 384])
y_data = np.copy(x_data)
x_bins = np.arange(384)
y_bins = np.linspace(0, 1, num=51)
edges = [16 * i for i in range(24)] + [15 + 16 * i for i in range(24)]
bool_ind = np.ones(384, dtype=bool)
bool_ind[edges] = 0

obs_ind = []
if args.obslist is not None:
    obslist = make_obslist(args.obslist)
for i, obs in enumerate(subdict.keys()):
    x_data[i] = np.arange(384)
    y_data[i] = subdict[obs]
    if args.obslist is not None:
        if obs in obslist:
            obs_ind.append(i)

if args.obslist is not None:
    bool_obs_ind = np.zeros(len(subdict.keys()), dtype=bool)
    bool_obs_ind[obs_ind] = 1

print(bool_obs_ind.shape)

plt.hist2d(x_data[:, bool_ind].flatten(), y_data[:, bool_ind].flatten(), bins=[x_bins, y_bins], norm=colors.LogNorm())
plt.title('2-d Long Run Occupation per Channel Histogram')
plt.xlabel('Channel #')
plt.ylabel('Occupation')
plt.colorbar(label='Counts')
plt.savefig('%s/2_d_occupation_per_chan_hist.pdf' % (outpath))
plt.close()

plt.hist(y_data[:, bool_ind].flatten(), bins=y_bins, histtype='step', label='Occupancy')
plt.title('1-d Long Run Occupation per Channel Histogram')
plt.xlabel('Occupation Fraction')
plt.ylabel('Counts')
plt.yscale('log')
plt.axvline(y_data[:, bool_ind].mean(), label='Mean Occupancy', color='black')
plt.axvline(np.sort(y_data[:, bool_ind].flatten())[-int(erfc(np.sqrt(2)) * y_data[:, bool_ind].size)], label='$2\sigma$', color='red')
plt.legend()
plt.savefig('%s/1_d_occupation_per_chan_hist.pdf' % (outpath))
plt.close()

plt.hist(y_data[:, bool_ind].mean(axis=1).flatten(), bins=y_bins, histtype='step', label='All Obs')
plt.hist(y_data[bool_obs_ind][:, bool_ind].mean(axis=1).flatten(), bins=y_bins, histtype='step', label='TV Subset')
plt.title('1-d Long Run Total Occupation Histogram')
plt.xlabel('Occupation Fraction')
plt.ylabel('Counts')
plt.yscale('log')
plt.axvline(y_data[:, bool_ind].mean(), label='Mean Occupancy', color='black')
plt.axvline(np.sort(y_data[:, bool_ind].mean(axis=1))[-int(erfc(np.sqrt(2)) * len(y_data))], label='$2\sigma$', color='red')
plt.legend()
plt.savefig('%s/1_d_occupation_total_hist.pdf' % (outpath))
plt.close()
