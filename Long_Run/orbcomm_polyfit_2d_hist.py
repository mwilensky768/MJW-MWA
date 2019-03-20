import numpy as np
import argparse
from SSINS import util, INS, plot_lib
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--obsfile')
parser.add_argument('-p', '--ppd_dir')
parser.add_argument('-i', '--ins_dir')
parser.add_argument('--outdir')
args = parser.parse_args()

obslist = util.make_obslist(args.obsfile)
missing_obs = util.make_obslist('/Users/mike_e_dubs/MWA/INS/Long_Run/Original_Jackknife_Revamp_Complete/missing_obs.txt')

maxlist = []
quadlist = np.zeros([1, 336])
linlist = np.zeros([1, 336])

for obs in obslist:
    if obs not in missing_obs:
        pow = np.load('%s/cfg-%s.npy' % (args.ppd_dir, obs))
        pow_max = np.amax(pow)

        if not os.path.exists('%s/%s_ms_poly_coeff_order_2_XX.npy' % (args.outdir, obs)):
            read_paths = util.read_paths_construct(args.ins_dir, 'original', obs, 'INS')
            ins = INS(read_paths=read_paths, order=2, coeff_write=True, outpath=args.outdir, obs=obs)
        coeff = np.load('%s/%s_ms_poly_coeff_order_2_XX.npy' % (args.outdir, obs))
        med = np.absolute(coeff[:-1])

        maxlist.append(pow_max)
        quadlist = np.vstack([quadlist, med[:1]])
        linlist = np.vstack([linlist, med[1:2]])

print('The minimum peak is %f' % min(maxlist))
print('The maximum peak is %f' % max(maxlist))
print('The minimum quad coeff is %f' % np.amin(quadlist))
print('The maximum quad coeff is %f' % np.amax(quadlist))
print('The minimum lin coeff is %f' % np.amin(linlist))
print('The maximum lin coeff is %f' % np.amax(linlist))
print(quadlist.shape)

max_bins = np.linspace(90, 120, num=11)
quad_bins = np.linspace(0, 0.0005, num=11)
lin_bins = np.linspace(0, 0.15, num=11)

quad_counts, max_bins, quad_bins = np.histogram2d(maxlist, quadlist, bins=10)
lin_counts, max_bins, lin_bins = np.histogram2d(maxlist, linlist, bins=10)

ticks = range(0, 10, 2)
quad_counts = np.flipud(quad_counts)
lin_counts = np.flipud(lin_counts)
xticklabels_quad = quad_bins[::2]
xticklabels_quad = ['%.4f' % ticklabel for ticklabel in xticklabels_quad]
xticklabels_lin = lin_bins[::2]
xticklabels_lin = ['%.2f' % ticklabel for ticklabel in xticklabels_lin]
yticklabels = max_bins[::2]
yticklabels = ['%.2f' % ticklabel for ticklabel in yticklabels]
yticklabels.reverse()

fig_quad, ax_quad = plt.subplots()
fig_lin, ax_lin = plt.subplots()

plot_lib.image_plot(fig_quad, ax_quad, quad_counts, xticks=ticks, xticklabels=xticklabels_quad,
                    yticklabels=yticklabels, yticks=ticks, title='Quad 2d Histogram',
                    ylabel='Peak Max', xlabel='Quad Coefficient', aspect=1)
plot_lib.image_plot(fig_lin, ax_lin, lin_counts, xticks=ticks, yticks=ticks,
                    xticklabels=xticklabels_lin, yticklabels=yticklabels, title='Lin 2d Histogram',
                    ylabel='Peak Max', xlabel='Lin Coefficient', aspect=1)
fig_quad.savefig('%s/quad_2dhist.png' % (args.outdir))
fig_lin.savefig('%s/lin_2dhist.png' % (args.outdir))
