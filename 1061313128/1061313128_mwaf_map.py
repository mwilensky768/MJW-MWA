from astropy.io import fits
import numpy as np

flag_arr_seq = []
for i in range(1, 25):
    if i < 10:
        i_str = '0%i' % i
    else:
        i_str = str(i)
    file = '/astro/mwaeor/MWA/data/1061313128/1061313128_%s.mwaf' % i_str
    hdulist = fits.open(file)
    nchan = hdulist[0].header['NCHANS']
    nant = hdulist[0].header['NANTENNA']
    ntime = hdulist[0].header['NSCANS']
    nbl = nant * (nant + 1) / 2
    flag_arr_seq.append(hdulist[1].data['FLAGS'].reshape([ntime, nbl, nchan]))
flag_map = np.mean(np.concatenate(flag_arr_seq, axis=2), axis=1)
np.save('/group/mwaeor/mwilensky/1061313128_flag_map.npy', flag_map)
