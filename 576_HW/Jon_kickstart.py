from SSINS import SS
import numpy as np

inpath = '/Users/mike_e_dubs/MWA/Data/uvfits/1061312272.uvfits'
read_kwargs = {'ant_str': 'cross'}

ss = SS(inpath=inpath, read_kwargs=read_kwargs)
counts, bins = np.histogram(ss.UV.data_array, bins='auto')
