import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors

path = '/Users/mike_e_dubs/MWA/INS/Long_Run/Original_Jackknife_Revamp_Complete/occ_dict.pik'
outpath = '/Users/mike_e_dubs/Documents/My_Papers/SSINS_Paper'

with open(path, 'rb') as file:
    occ_dict = pickle.load(file)

subdict = occ_dict[5]['total']
x_data = np.zeros([len(subdict), 384])
y_data = np.copy(x_data)
x_bins = np.arange(384)
y_bins = np.linspace(0, 1, num=51)
edges = [16 * i for i in range(24)] + [15 + 16 * i for i in range(24)]
bool_ind = np.ones(384, dtype=bool)
bool_ind[edges] = 0

for i, obs in enumerate(subdict):
    x_data[i] = np.arange(384)
    y_data[i] = subdict[obs]

plt.hist2d(x_data[:, bool_ind].flatten(), y_data[:, bool_ind].flatten(), bins=[x_bins, y_bins], norm=colors.LogNorm())
plt.title('2-d Long Run Occupation per Channel Histogram')
plt.xlabel('Channel #')
plt.ylabel('Occupation')
plt.colorbar(label='Counts')
plt.savefig('%s/2_d_occupation_per_chan_hist.pdf' % outpath)
plt.close()

plt.hist(y_data[:, bool_ind].flatten(), bins=y_bins)
plt.title('1-d Long Run Occupation per Channel Histogram')
plt.xlabel('Occupation Fraction')
plt.ylabel('Counts')
plt.yscale('log')
plt.savefig('%s/1_d_occupation_per_chan_hist.pdf' % outpath)
plt.close()
