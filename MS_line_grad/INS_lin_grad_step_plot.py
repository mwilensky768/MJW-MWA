import argparse
from SSINS import util
from SSINS import INS
from SSINS import MF
from SSINS import Catalog_Plot as cp
from matplotlib import cm

parser = argparse.ArgumentParser()
parser.add_argument('basedir')
parser.add_argument('outdir')
parser.add_argument('obsfile')
parser.add_argument('flag_choice')
parser.add_argument('order', type=int)
parser.add_argument('--labels', nargs='*')
parser.add_argument('--mins', nargs='*', type=float)
parser.add_argument('--maxs', nargs='*', type=float)
args = parser.parse_args()

if args.labels is not None:
    shape_dict = {label: [min, max] for (label, min, max) in zip(args.labels, args.mins, args.maxs)}
else:
    shape_dict = shape_dict

obslist = util.make_obslist(args.obsfile)

for obs in obslist:
    read_paths = util.read_paths_INS(args.basedir, args.flag_choice, obs)
    ins = INS(read_paths=read_paths, flag_choice=args.flag_choice, obs=obs,
              outpath=args.outdir, order=args.order)
    cp.INS_plot(ins)
    mf = MF(ins, sig_thresh=5, shape_dict=shape_dict)
    mf.apply_match_test(order=args.order)
    cp.MF_plot(mf)
