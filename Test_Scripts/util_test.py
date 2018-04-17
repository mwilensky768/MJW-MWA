import numpy as np
from matplotlib import use, cm
use('Agg')
import matplotlib.pyplot as plt
import rfiutil as rfiu
import plot_lib

with open('/Users/mike_e_dubs/MWA/Obs_Lists/P2_Streak.txt') as f:
    obslist = f.read().split("\n")

arr_path = '/Users/mike_e_dubs/MWA/Catalogs/Wenyang_Phase2/data_eva/arrs/arrs/'
outpath = '/Users/mike_e_dubs/MWA/Test_Plots/streak_test/'
freq_arr = np.load('/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/MWA_Highband_Freq_Array.npy')

for obs in obslist:
    frac_diff = np.load('%s%s_Unflagged_Amp_INS_frac_diff.npym' % (arr_path, obs))

    fig, ax = plt.subplots(figsize=(14, 8))
    fig_im, ax_im, pols, xticks, xminors, yminors, xticklabels = plot_lib.four_panel_tf_setup(freq_arr)
    fig_im.suptitle('%s Streak Flagging Test' % (obs))

    frac_diff = rfiu.streak_detect(frac_diff)

    # plot_lib.line_plot(fig, ax, [edge[:, 0, m] for m in range(edge.shape[2])],
                       # title='%s Streak Detection' % (obs), xlabel='Time', ylabel='Gradient',
                       # labels=['XX', 'YY', 'XY', 'YX'])
    for m in range(frac_diff.shape[3]):
        plot_lib.image_plot(fig_im, ax_im[m / 2][m % 2], frac_diff[:, 0, :, m],
                            cmap=cm.coolwarm, xlabel='Frequency (Mhz)',
                            ylabel='Time Pair', cbar_label='Amplitude (UNCALIB)',
                            zero_mask=False, mask_color='black',
                            title=pols[m], xticks=xticks, xticklabels=xticklabels,
                            xminors=xminors, yminors=yminors)

    fig_im.savefig('%s%s_streak_detect.png' % (outpath, obs))
    #fig.savefig('%s%s_edge_detect.png' % (outpath, obs))
    plt.close(fig)
    plt.close(fig_im)
