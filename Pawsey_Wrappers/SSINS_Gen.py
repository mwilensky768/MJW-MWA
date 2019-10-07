from __future__ import absolute_import, division, print_function

import argparse
from SSINS import Catalog_Plot as cp
from SSINS import SS, INS
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('obs', action='store', help='The observation ID')
parser.add_argument('inpath', action='store', help='The path to the data file')
parser.add_argument('outpath', action='store', help='The base directory for saving all outputs')
args = parser.parse_args()

ss = SS()
ss.read(args.inpath, ant_str='cross')

ins = INS(ss)

prefix = '%s/%s' % (args.outpath, args.obs)
ins.write(prefix)
cp.INS_plot(ins, prefix)
