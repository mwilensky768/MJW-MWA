import numpy as np
import SSINS.util as u
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--obsfile')
parser.add_argument('-i', '--indir')
parser.add_argument('-t', '--thresh', type=float)
parser.add_argument('-g', action='store_true')
parser.add_argument('--filter', action='store_true')
parser.add_argument('-w', type=int)
args = parser.parse_args()

if args.g:
    prefix = 'cfg-'
else:
    prefix = 'cf-'

obslist = u.make_obslist(args.obsfile)
sublist = []

for obs in obslist:
    pow = np.load('%s/%s%s.npy' % (args.indir, prefix, obs))
    if not args.filter:
        cond = np.any(pow > args.thresh)
    else:
        pow_ds = np.zeros((2, 256 / args.w))
        for i in range(2):
            for k in range(256 / args.w):
                pow_ds[i, k] = np.amax(pow[i, args.w * k: args.w * (k + 1)])
        pow_ds = pow_ds[:, 10:20]
        cond = np.any(np.logical_and(pow_ds[:, 1:-1] > pow_ds[:, :-2] + args.thresh,
                                     pow_ds[:, 1:-1] > pow_ds[:, 2:] + args.thresh))
    if cond:
        sublist.append(obs)

print('The length of sublist is %i' % len(sublist))
if not args.filter:
    u.make_obsfile(sublist, '%s/%s_t%s.txt' % (args.indir, prefix, args.thresh))
else:
    u.make_obsfile(sublist, '%s/%s_filter_w%s_t%s_b10_20.txt' % (args.indir, prefix, args.w, args.thresh))
