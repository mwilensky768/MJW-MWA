from SSINS import SS
from SSINS import Catalog_Plot as cp

inpath = '/Volumes/Faramir/uvfits'
obslist = ['1061313128', '1061312640']
outpath = '/Users/mikewilensky/General/Hist_Compare_Big_Legend'
read_kwargs = {'ant_str': 'cross',
               'bad_time_indices': [0, -1, -2, -3]}

for obs in obslist:
    ss = SS(inpath='%s/%s.uvfits' % (inpath, obs), obs=obs, outpath=outpath,
            read_kwargs=read_kwargs)
    ss.VDH_prepare()
    cp.VDH_plot(ss.VDH, leg_size='xx-large')
    del ss
