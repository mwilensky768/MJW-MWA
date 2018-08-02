import numpy as np
from SSINS import INS
from SSINS import MF
import glob

"""
Read in some pre-filtered noise spectra and make a scatter plot of their RFI
occupancies as a function of LST
"""

parser = argparse.ArgumentParser()
INS_inpath = '/Users/mike_e_dubs/MWA/INS/Long_Run'
parser.add_argument('outpath', action='store', help='The output base directory')

args = parser.parse_args()

pointings = ['Minus_1', 'Minus_2', 'Plus_1', 'Plus_2', 'Zenith']
obslists = glob.glob('/Users/mike_e_dubs/python_stuff/MJW-MWA/Obs_Lists/beardsley*')
obslists.sort()

for pointing, obslist in zip(pointings, obslists):
    with open(obslist) as f:
        obslist = f.read().split("\n")
    obslist.remove('')
    for obs in obslist:
        database = '%s/%s_Match_Filter_Only' % (INS_inpath, pointing)
