from SSINS import util
from SSINS import INS
from SSINS import Catalog_Plot as cp
from SSINS import MF
import numpy as np
import os

obslist_path = '/Users/mike_e_dubs/HERA/Interesting_Obs_Master.txt'
obslist = util.make_obslist(obslist_path)
basedirs = ['/Users/mike_e_dubs/HERA/INS/IDR2_Prelim_Nocut/HERA_IDR2_Prelim_Set_nocut',
            '/Users/mike_e_dubs/HERA/INS/IDR2_OR/HERA_IDR2_Prelim_Set_OR_original']

shape_dict = {'dig1': [1.125e8, 1.15625e8],
              'dig2': [1.375e8, 1.40625e8],
              'dig3': [1.625e8, 1.65625e8],
              'dig4': [1.875e8, 1.90625e8],
              'TV4': [1.74e8, 1.82e8],
              'TV5': [1.82e8, 1.9e8],
              'TV6': [1.9e8, 1.98e8],
              'TV7': [1.98e8, 2.06e8]}


for i, flag_choice in enumerate(['None', 'original']):
    for obsid in obslist:
        for pol in ['xx', 'yy', 'xy', 'yx']:
            obs = 'zen.%s.%s.HH' % (obsid, pol)
            read_paths = util.read_paths_INS(basedirs[i], flag_choice, obs)
            ins = INS(read_paths=read_paths, obs=obs, flag_choice=flag_choice,
                      outpath='/Users/mike_e_dubs/HERA/INS/Adam_Presentation/%s' % flag_choice)
            cp.INS_plot(ins, vmax=0.05, ms_vmax=5, ms_vmin=-5)
            if flag_choice is 'None':
                
