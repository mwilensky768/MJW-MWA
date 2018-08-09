import shutil
import glob
import numpy as np

png_base = '/Users/mike_e_dubs/MWA/TV_Images/Sidelobe_Hunt'
INS_base = '/Users/mike_e_dubs/MWA/INS/Long_Run/All/Match_Filter/figs'

for dir in ['Primary', 'Second_Southern', 'Unidentified']:
    obslist = glob.glob('%s/%s/*' % (png_base, dir))
    L = len(png_base) + len(dir) + 2
    obslist = np.unique([obs[L:L + 10] for obs in obslist])
    for obs in obslist:
        for plt in ['', '_ms', '_ms_match']:
            shutil.copy('%s/%s_spw0_INS%s.png' % (INS_base, obs, plt), '%s/%s' % (png_base, dir))
