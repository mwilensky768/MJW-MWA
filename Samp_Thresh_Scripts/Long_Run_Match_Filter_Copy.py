import shutil
import glob

basepath = '/Users/mike_e_dubs/MWA/INS/Long_Run'
for name in ['Minus_1', 'Minus_2', 'Plus_1', 'Plus_2', 'Zenith']:
    obslist = glob.glob('%s/%s_Samp_Thresh/*spw0_MS.png' % (basepath, name))
    L = len('%s/%s_Samp_Thresh/' % (basepath, name))
    obslist = [path[L:L + 10] for path in obslist]
    for obs in obslist:
        shutil.copy('%s/%s_Match_Filter_Only/figs/%s_spw0_INS_ms_match.png' %
                    (basepath, name, obs), '%s/%s_Samp_Thresh' % (basepath, name))
