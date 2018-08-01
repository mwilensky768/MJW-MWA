import numpy as np
import glob

keys = ['Minus_1', 'Minus_2', 'Plus_1', 'Plus_2', 'Zenith']
base = '/Users/mike_e_dubs/MWA/INS/Long_Run/time_arrs'
JD = np.array([])
for i, key in enumerate(keys):
    pathlist = glob.glob('%s/%s/*times_arr.npy' % (base, key))
    for path in pathlist:
        JD = np.append(JD, np.load(path)[0])
JD.sort()
print('JD max is %f' % JD[-1])
print('JD min is %f' % JD[0])
print('JD range is %f' % (JD[-1] - JD[0]))
diffs = np.diff(JD)
print('Infimum of time differences greater than a day is %f' % diffs[diffs > 1].min())
print('Infimum of time differences greater than 12 hours is %f' % diffs[diffs > 0.5].min())
print('Infimum of time differences greater than 6 hours is %f' % diffs[diffs > 0.25].min())
print('The number of days in this subset of the long run is %f' % len(diffs[diffs > 1] + 1))
print(np.unique(np.floor(JD)))
print(JD[np.where(diffs > 1)])
