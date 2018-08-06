import numpy as np

p_obs = [1061313616, 1061315448, 1061317272, 1061319472]

for obs in p_obs:
    lst_p = np.load('/Users/mike_e_dubs/MWA/INS/Long_Run/All/metadata/%i_lst_array.npy' % (obs))
    if lst > np.pi:
        lst_p -= 2 * np.pi
    lst_p *= 23.9345 / (2 * np.pi)

print()
