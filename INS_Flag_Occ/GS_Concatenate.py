import numpy as np
from SSINS import util
import glob

basedir = '/Users/mike_e_dubs/MWA/INS/Golden_Set_80khz_0.5s'
obslist_dir
suff = ['0_56', '56_112', '112_168', '168_224']
L = len('%s/GS_SSINS_0_56/' % basedir)
pathlist = glob.glob('%s/GS_SSINS_0_56/*data.npy')
obslist = []
for path in pathlist:
    obslist.append(path[L:L + 10])

for obs in obslist:
    data = np
for i in np.arange(4):
