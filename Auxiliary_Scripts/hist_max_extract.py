import numpy as np
import glob
from matplotlib import cm, use
use('Agg')
import matplotlib.pyplot as plt
import plot_lib

arrpath = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Golden_Set_8s_Autos/Freq_Time/Variances/'
outpath = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Golden_Set_8s_Autos/Freq_Time/max_loc/'
hist_arrs_all = glob.glob('%s*All_hist.npy' % (arrpath))
hist_arrs_all.sort()
hist_arrs_unflagged = glob.glob('%s*Unflagged_hist.npy' % (arrpath))
hist_arrs_unflagged.sort()
bins_arrs_all = glob.glob('%s*All_bins.npy' % (arrpath))
bins_arrs_all.sort()
bins_arrs_unflagged = glob.glob('%s*Unflagged_bins.npy' % (arrpath))
bins_arrs_unflagged.sort()
print('There are %i obs to process ' % (len(hist_arrs_all)))
sigmas = glob.glob('%s*sigma*' % (arrpath))

max_locs = []
for (hist_all_arr, hist_unflagged_arr, bins_all_arr, bins_unflagged_arr) in \
        zip(hist_arrs_all, hist_arrs_unflagged, bins_arrs_all, bins_arrs_unflagged):

    obs1 = hist_all_arr[len(arrpath):len(arrpath) + 10]
    obs2 = hist_unflagged_arr[len(arrpath):len(arrpath) + 10]
    obs3 = bins_all_arr[len(arrpath):len(arrpath) + 10]
    obs4 = bins_unflagged_arr[len(arrpath):len(arrpath) + 10]
    assert obs1 == obs2
    assert obs2 == obs3
    assert obs3 == obs4

    hist_all = np.load(hist_all_arr)
    hist_unflagged = np.load(hist_unflagged_arr)
    bins_all = np.load(bins_all_arr)
    bins_unflagged = np.load(bins_unflagged_arr)

    bins_all_widths = np.diff(bins_all)
    bins_unflagged_widths = np.diff(bins_unflagged)
    bins_all_centers = bins_all[:-1] + 0.5 * bins_all_widths
    bins_unflagged_centers = bins_unflagged[:-1] + 0.5 * bins_unflagged_widths

    max_loc_all = bins_all_centers[hist_all.argmax()]
    max_loc_unflagged = bins_unflagged_centers[hist_unflagged.argmax()]

    max_locs.append([int(obs1), max_loc_unflagged, max_loc_all])

max_locs.sort()
max_locs = np.array(max_locs)
fig, ax = plt.subplots(figsize=(14, 8), nrows=2)
fig.suptitle('Golden Set Histogram Count Max Locations')
fig_titles = ['Unflagged', 'All']
xticks = [1061313496, 1061315320, 1061317152, 1061318984]

for m in range(2):
    plot_lib.scatter_plot_2d(fig, ax[m], max_locs[:, 0], max_locs[:, m + 1],
                             title=fig_titles[m], xlabel='GPS Time',
                             ylabel='Max Location (UNCALIB)', xticks=xticks)

np.save('%smax_locs.npy' % (outpath), max_locs)
fig.savefig('%smax_locs_day1.png' % (outpath))
