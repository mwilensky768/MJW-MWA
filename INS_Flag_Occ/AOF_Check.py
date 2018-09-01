from __future__ import division

import numpy as np
import glob


indir = '/Users/mike_e_dubs/MWA/INS/Golden_Set/AO_Sanity_Check'
num_arrs = glob.glob('%s/*num.npy' % indir)
den_arrs = glob.glob('%s/*den.npy' % indir)
num_arrs.sort()
den_arrs.sort()

num = 0
den = 0

for num_arr, den_arr in zip(num_arrs, den_arrs):
    num += np.load(num_arr)
    den += np.load(den_arr) * 0.875
total_occ = num / den * 100
print('The RFI occupancy of the Golden Set according to COTTER is %f%%' % total_occ)
