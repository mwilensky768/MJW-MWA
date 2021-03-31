from SSINS import SS, INS, util
from SSINS import Catalog_Plot as cp
import argparse
import numpy as np
import os


def get_freq_ticks(freq_array, freqlist):
    xticks = []
    xticklabels = []
    for freq in freqlist:
        xticks.append(np.argmin(np.abs(freq_array - freq)))
        xticklabels.append(str(int(freq * 10 ** (-6))))
    return(xticks, xticklabels)


parser = argparse.ArgumentParser()
parser.add_argument('obslist')
args = parser.parse_args()

obslist = util.make_obslist(args.obslist)

obsdir = '/Volumes/Faramir/uvfits'
outdir = '/Users/mikewilensky/2016_funny_flag'

freqlist = np.arange(170, 200, 5) * 10**6

for obs in obslist:
    filepath = '%s/%s_uvfits/%s.uvfits' % (obsdir, obs, obs)
    prefix = '%s/%s' % (outdir, obs)
    outpath = '%s_SSINS_data.h5' % prefix
    if not os.path.exists(outpath):
        ss = SS()
        ss.read(filepath, read_data=False)
	ss.read(filepath, times=np.unique(ss.time_array)[2:-3])
        ins = INS(ss)
	ins.write(prefix)
    else:
        ins = INS(outpath)
    xticks, xticklabels = get_freq_ticks(ins.freq_array, freqlist)
    print(ins.freq_array[1] - ins.freq_array[0])
    cp.INS_plot(ins, '%s_SSINS.pdf' % prefix, xticks=xticks,
                xticklabels=xticklabels)
