import numpy as np
import glob

for fn in ['Minus_1', 'Minus_2', 'Zenith', 'Plus_1', 'Plus_2']:
    path = '/Users/mike_e_dubs/MWA/INS/Long_Run/%s_Samp_Thresh' % fn
    L = len(path) + 1
    obslist = glob.glob('%s/*.png' % path)
    obslist = np.unique([path[L:L + 10] for path in obslist])
    f = open('/Users/mike_e_dubs/python_stuff/MJW-MWA/Obs_Lists/Long_Run_Samp_Thresh_%s.txt' % fn, 'w')
    for obs in obslist:
        f.write('%s\n' % obs)
