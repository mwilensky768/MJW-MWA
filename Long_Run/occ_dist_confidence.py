from __future__ import division

import numpy as np
import pickle
from SSINS import plot_lib
import matplotlib.pyplot as plt
from scipy.special import erf
import shutil
import os
from SSINS import util
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('shape')
args = parser.parse_args()

occ_dict_path = '/Users/mike_e_dubs/MWA/INS/Long_Run/Original_Jackknife/long_run_original_occ_dict.pik'
ins_plots = '/Users/mike_e_dubs/MWA/INS/Long_Run/Original/figs'
outdir = '/Users/mike_e_dubs/MWA/INS/Long_Run/Original_Jackknife/2_sig_figs_%s' % args.shape
if not os.path.exists(outdir):
    os.makedirs(outdir)
occ_dict = pickle.load(open(occ_dict_path, 'rb'))
subdict = occ_dict[5][args.shape]

edges = [16 * k for k in range(24)] + [15 + 16 * k for k in range(24)]
good_freqs = np.ones(384, dtype=bool)
good_freqs[edges] = 0

bins = np.linspace(0, 1, num=51)

occlist = []
obslist = []
for obs in subdict:
    if args.shape == 'total':
        total_occ = np.mean(subdict[obs][good_freqs])
    else:
        total_occ = subdict[obs]
    occlist.append(total_occ)
    if total_occ > 0.5:
        obslist.append(obs)
        shutil.copy('%s/%s_original_INS_data.png' % (ins_plots, obs), outdir)
counts, bins = np.histogram(occlist, bins=bins)
cdf = np.cumsum(counts)
cdf = cdf / cdf[-1]
inds = np.array([np.where(cdf > erf(k / np.sqrt(2)))[0][0] for k in range(1, 4)])
bin_edges = bins[inds + 1]
print('The 1 sigma, 2 sigma, and 3 sigma occupations are %s' % bin_edges)
print(obslist)
util.make_obsfile(obslist, '%s/2_sig_obs.txt' % outdir)
