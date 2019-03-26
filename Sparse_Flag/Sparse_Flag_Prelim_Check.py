from SSINS import SS, INS, util
from SSINS import Catalog_Plot as cp
import argparse
import numpy as np


def get_freq_xticks(freq_array, freqlist):
    xticks = []
    xticklabels = []
    for freq in freqlist:
        xticks.append(np.argmin(np.abs(freq_array - freq)))
        xticklabels.append(str(int(freq)))
    return(xticks, xticklabels)


parser = argparse.ArgumentParser()
parser.add_argument('obslist')
args = parser.parse_args()

obslist = util.make_obslist(args[0])

obsdir = '/Volumes/Faramir/uvfits'
outdir = '/Users/mikewilensky/2016_funny_flag'

freqlist = np.arange(170, 200, 5) * 10**6

for obs in obslist:
    filepath = '%s/%s_uvfits/%s.uvfits' % (obsdir, obs, obs)
    prefix = '%s/%s' % (outdir, obs)
    outpath = '%s_SSINS_data.h5' % prefix
    if not os.path.exists(outpath):
        ss = SS()
        ss.read(filepath)
        ins = INS(ss)
    else:
        ins = INS(outpath)
    xticks, xticklabels = get_freq_ticks(ins.freq_array, freqlist)
    cp.INS_plot(ins, '%s_SSINS.pdf' % prefix, xticks=xticks,
                xticklabels=xticklabels, xlabel='Frequency (Mhz)',
                ylabel='Times (2 s)')
