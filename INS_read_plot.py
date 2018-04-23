from matplotlib import cm, use
use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import plot_lib
import os
import argparse

"""
A function which returns an INS and its mean subtracted form
[numpy array of dim Ntimes, 1, Nfreqs, Npols]. Must provide and input filepath,
output directory, obsid, and the directory that points to this repository.

Can also run from the command line if you provide an input filepath that points to
a saved numpy array and an output directory to save a figure and an array.
This will just save a plot and a numpy array and then exit python.

ALL PATHS TO DIRECTORIES SHOULD END IN "/" OTHERWISE THERE WILL BE A MESS
"""


def INS_read_plot(inpath, outpath, obs, repo_dir):

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    fig_titles = {'All': 'All Baselines', 'Unflagged': 'Post-Flagging'}
    for flag in fig_titles:
        if flag in inpath:
            fig_title = fig_titles[flag]
            flag_slice = flag

    INS = np.load(inpath)
    INS_frac_diff = INS / INS.mean(axis=1) - 1

    fig, ax, pols, xticks, xminors, yminors, xticklabels = \
        plot_lib.four_panel_tf_setup(np.load('%sUseful_Information/MWA_Highband_Freq_Array.npy' %
                                             (repo_dir)))
    fig.suptitle('%s Mean-Subtracted Incoherent Noise Spectrum, %s' % (obs, fig_title))

    for m, pol in enumerate(pols):
        plot_lib.image_plot(fig, ax[m / 2][m % 2], INS_frac_diff[:, 0, :, m],
                            cmap=cm.coolwarm, title=pol, cbar_label='Fraction of Mean',
                            xticks=xticks, xminors=xminors, yminors=yminors,
                            xticklabels=xticklabels, zero_mask=False,
                            mask_color='black')

    fig.savefig('%s%s_INS_frac_diff_%s.png' % (outpath, obs, flag_slice))
    np.save('%s%s_INS_frac_diff_%s.npy' % (outpath, obs, flag_slice), INS_frac_diff)

    return(INS, INS_frac_diff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inpath", action='store', nargs=1, help="The file you want to process")
    parser.add_argument("outpath", action='store', nargs=1,
                        help="The target directory for plots and arrays, be sure to include the final /")
    args = parser.parse_args()
    for n in range(0, len(args.inpath[0]) - 10):
        try:
            int(args.inpath[0][n:n + 10])
            obs = args.inpath[0][n:n + 10]
        except:
            pass
    repo_dir = os.getcwd() + '/'

    INS, INS_frac_diff = INS_read_plot(args.inpath[0], args.outpath[0], obs, repo_dir)
