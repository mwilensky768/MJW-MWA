import glob
import numpy as np

indir = '/Users/mike_e_dubs/MWA/INS/Long_Run/All/metadata'
lr_list_path = '/Users/mike_e_dubs/python_stuff/MJW-MWA/Obs_Lists'
obslist = glob.glob('%s/*lst_array*' % (indir))
L = len(indir) + 1
obslist = [path[L:L + 10] for path in obslist]
print(len(obslist))
with open('%s/Long_Run_8s_Autos_OBSIDS.txt' % lr_list_path) as f:
    lr_list = f.read().split("\n")
lr_list.remove('')
print(len(lr_list))
print(len(np.unique(lr_list)))
final_list = [obs for obs in lr_list if obs not in obslist]
print(len(final_list))
g = open('%s/Long_Run_Missed_Obs.txt' % lr_list_path, 'w')
for obs in final_list:
    g.write('%s\n' % obs)
