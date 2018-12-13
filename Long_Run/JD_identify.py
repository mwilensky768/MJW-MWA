from __future__ import division

import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('date', type=int)
args = parser.parse_args()

indir = '/Users/mike_e_dubs/MWA/INS/Long_Run/time_arrs'
JD_paths = glob.glob('%s/*times*' % indir)
obslist = []

for path in JD_paths:
    obs = path[len(indir) + 1:len(indir) + 11]
    date = np.load(path)[0]
    if int(date) == args.date:
        obslist.append(obs)

print(len(obslist))
print(obslist)
