import shutil
import numpy as np
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('rawdir', help='The directory with the raw plots')
parser.add_argument('outdir', help='The target directory for copied files')
args = parser.parse_args()
plots = glob.glob('%s/*.png' % (args.outdir))
L = len(args.outdir)
obsids = np.unique([plot[L:L + 10] for plot in plots])
for obs in obsids:
    raw_plots = glob.glob('%s/%s*.png' % (args.rawdir, obs))
    for raw_plot in raw_plots:
        shutil.copy(raw_plot, args.outdir)
