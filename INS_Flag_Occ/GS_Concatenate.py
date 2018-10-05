import numpy as np
from SSINS import util
import glob

basedir = '/Users/mike_e_dubs/MWA/INS/Golden_Set_80khz_0.5s'
suff = ['0_96', '96_192', '192_288', '288_384']
L = len('%s/GS_SSINS_0_96_80khz/arrs/' % basedir)
pathlist = glob.glob('%s/GS_SSINS_0_96_80khz/arrs/*data.npym' % basedir)

obslist = []
for path in pathlist:
    obslist.append(path[L:L + 10])

for obs in obslist:
    for i in np.arange(4):
        data_path = '%s/GS_SSINS_%s_80khz/arrs/%s_None_INS_data.npym' % (basedir, suff[i], obs)
        Nbls_path = '%s/GS_SSINS_%s_80khz/arrs/%s_None_INS_Nbls.npym' % (basedir, suff[i], obs)
        if not i:
            data = np.load(data_path)
            Nbls = np.load(Nbls_path)
        else:
            data = np.concatenate((data, np.load(data_path)), axis=2)
            Nbls = np.concatenate((Nbls, np.load(Nbls_path)), axis=2)
    np.save('%s/GS_SSINS_concat/arrs/%s_None_INS_data.npym' % (basedir, obs), data)
    np.save('%s/GS_SSINS_concat/arrs/%s_None_INS_Nbls.npym' % (basedir, obs), Nbls)
