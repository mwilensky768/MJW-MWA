import numpy as np
import glob

basedir = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Sumit_Catalog/Notable_Cases/'
subdir_dict = {'DS_2015/': ['Clip/', 'Iso_Broadband/', 'Saturated/', 'Striations/'],
               'LR_2013/': ['Clip/', ], 'S2_2014/': ['Dip/', 'Striations/']}

fil = open('/Users/mike_e_dubs/MWA/Obs_Lists/PPD_Obs.txt', 'w')

for subdir in subdir_dict:
    for typ in subdir_dict[subdir]:
        obslist = np.array(glob.glob('%s%s%s*.png' % (basedir, subdir, typ)))
        obslist = obslist[range(0, len(obslist), 2)]
        obs_start = len(basedir) + len(subdir) + len(typ)
        obs_end = obs_start + 10
        obslist = [obs[obs_start:obs_end] for obs in obslist]
        for obs in obslist:
            fil.write(obs + '\n')
