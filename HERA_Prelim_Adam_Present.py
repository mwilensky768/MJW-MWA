from SSINS import util
from SSINS import INS
import numpy as np

obslist_path = '/Users/mike_e_dubs/HERA/Interesting_Obs_Master.txt'
obslist = util.make_obslist(obslist_path)

for obs in obslist:
    obs = 'zen.%s.xx.HH' % obs
    read_paths = util.read_paths_INS()
    ins = INS
    for pol in ['yy', 'xy', 'yx']:
