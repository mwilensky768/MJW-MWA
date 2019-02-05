from astropy.io import fits
import numpy as np

flag_arr_seq = []
min_ntime = 224
for i in range(1, 25):
    if i < 10:
        i_str = '0%i' % i
    else:
        i_str = str(i)
    file = '/Users/mike_e_dubs/MWA/Data/mwafs/1061312760_%s.mwaf' % i_str
    hdulist = fits.open(file)
    nchan = hdulist[0].header['NCHANS']
    nant = hdulist[0].header['NANTENNA']
    ntime = len(hdulist[1].data['FLAGS']) / 8256
    if ntime < min_ntime:
        min_ntime = ntime
    nbl = nant * (nant + 1) / 2
    flag_arr_seq.append(hdulist[1].data['FLAGS'].reshape([ntime, nbl, nchan])[:min_ntime])
flag_map = np.mean(np.concatenate(flag_arr_seq, axis=2), axis=1)
np.save('/Users/mike_e_dubs/General/1061312760_flag_map.npy', flag_map)
