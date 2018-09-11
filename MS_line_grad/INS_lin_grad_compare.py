import argparse
from SSINS import util
from SSINS import INS
from SSINS import Catalog_Plot as cp
from matplotlib import cm

parser = argparse.ArgumentParser()
parser.add_argument('basedir')
parser.add_argument('outdir')
parser.add_argument('obsfile')
parser.add_argument('flag_choice')
args = parser.parse_args()

obslist = util.make_obslist(args.obsfile)

for obs in obslist:
    read_paths = util.read_paths_INS(args.basedir, args.flag_choice, obs)
    ins = INS(read_paths=read_paths, flag_choice=args.flag_choice, obs=obs,
              outpath=args.outdir)
    ins.data = ins.mean_subtract(order=1)
    cp.INS_plot(ins, ms_vmax=5, ms_vmin=-5, vmin=-5, vmax=5,
                data_cmap=cm.coolwarm)
