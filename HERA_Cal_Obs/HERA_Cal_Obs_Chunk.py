from pyuvdata import UVData
import numpy as np
from SSINS import util as u
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('obsfile')
parser.add_argument('indir')
parser.add_argument('outdir')
args = parser.parse_args()

obslist = u.make_obslist(args.obsfile)

for obs in obslist:
    UV = UVData()
    UV.read('%s/%s.uvfits' % (args.indir, obs), file_type='uvfits')
    for chunk in range(12):
        times = np.unique(UV.time_array)[5 * chunk: 5 * (chunk + 1)]
        UV1 = UV.select(times=times, inplace=False)
        UV1.write_uvfits('%s/%s_%i.uvfits' % (args.outdir, obs, chunk + 1))
        del UV1
    del UV
