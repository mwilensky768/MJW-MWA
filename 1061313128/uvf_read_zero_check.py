from scipy.io import readsav
import numpy as np
import glob

indir = '/Users/mike_e_dubs/MWA/PS/SSINS'
uvf_list = glob.glob('%s/*weights*uvf*.idlsave' % indir)

for uvf_path in uvf_list:
    uvf = readsav(uvf_path)
    for attr in uvf:
        print(attr)
    where = np.where(uvf['variance_cube'] > 1)
    print(where)
