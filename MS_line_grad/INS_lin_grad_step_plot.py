import argparse
from SSINS import util
from SSINS import INS
from SSINS import MF
from SSINS import Catalog_Plot as cp
from SSINS import plot_lib
import matplotlib.pyplot as plt
import numpy as np
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
    shape_dict = {}

obslist = util.make_obslist(args.obsfile)
edges = [0 + 16 * i for i in range(24)] + [15 + 16 * i for i in range(24)]
freqs = np.load('/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/MWA_Highband_Freq_Array.npy')

for i, obs in enumerate(obslist):
    read_paths = util.read_paths_INS(args.basedir, args.flag_choice, obs)
    ins = INS(read_paths=read_paths, flag_choice=args.flag_choice, obs=obs,
              outpath=args.outdir, order=args.order)
    cp.INS_plot(ins)
    mf = MF(ins, sig_thresh=5, shape_dict=shape_dict)
    mf.apply_match_test(order=args.order)
    cp.MF_plot(mf)
    if not i:
        occ_num = np.ma.masked_array(mf.INS.data.mask)
        occ_den = np.ma.masked_array(np.ones(occ_num.shape))
        occ_num[:, 0, edges] = np.ma.masked
        occ_den[:, 0, edges] = np.ma.masked
    else:
        occ_num = occ_num + mf.INS.data.mask
        occ_den = occ_den + np.ones(occ_num.shape)

occ_freq = occ_num.sum(axis=(0, 1, 3)) / occ_den.sum(axis=(0, 1, 3)) * 100
occ_total = occ_num.sum() / occ_den.sum() * 100
fig, ax = plt.subplots(figsize=(14, 8))
plot_lib.error_plot(fig, ax, freqs * 10**(-6), occ_freq,
                    title='Golden Set RFI Frequency Post COTTER',
                    xlabel='Frequency (Mhz)', ylabel='RFI Occupancy %%',
                    legend=False)
fig.savefig('%s/figs/RFI_Occupancy.png' % args.outdir)
print('The total RFI occupancy is %s %%' % (occ_total))
