from SSINS import SS
from SSINS import Catalog_Plot as cp
import numpy as np

inpath = '/Volumes/Faramir/uvfits'
obslist = ['1061313128', '1061312640', '1066742016']
outpath = '/Users/mikewilensky/General/Hist_Compare_Big_Legend'
read_kwargs = {'ant_str': 'cross'}

for obs in obslist:
    ss = SS(inpath='%s/%s.uvfits' % (inpath, obs), obs=obs, outpath=outpath,
            read_kwargs=read_kwargs, bad_time_indices=[0, -1, -2, -3])
    ss.VDH_prepare(fit_hist=True, bins='auto')
    cp.VDH_plot(ss.VDH, leg_size='xx-large', xscale='linear', ylim=[0.1, 10 * np.amax(ss.VDH.counts[0])])
    del ss
