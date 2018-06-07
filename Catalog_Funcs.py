import numpy as np
from matplotlib import use
use('Agg')
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import plot_lib
import scipy.linalg
import rfiutil
import os
import itertools


def band_constructor(counts, bins, labels, flag_slice):

    # Find which label belongs to the flag slice for which the band is constructed
    for k, label in enumerate(labels):
        if label == flag_slice:
            m = k

    # Identify the corresponding counts and corresponding fit
    count = counts[m]
    fit = counts[m + 1]

    # Find the first left bin edge where the counts have their maximum
    max_loc = bins[:-1][count.argmax()]
    # The band starts at the minimum left bin edge for which the fit < 1 and bins > max_loc,
    # and ends at the right-most bin-edge.
    band = [min(bins[:-1][np.logical_and(fit < 1, bins[:-1] > max_loc)]), max(bins)]

    return(band)


def ext_list_selector(RFI, W):

    if RFI.UV.Npols == 4:
        MAXW_list = range(4)
        MAXW_list[:2] = [max([np.amax(W[:, :, l]) for l in [0, 1]]) for m in [0, 1]]
        MAXW_list[2:4] = [max([np.amax(W[:, :, l]) for l in [2, 3]]) for m in [0, 1]]

        MINW_list = range(4)
        MINW_list[:2] = [min([np.amin(W[:, :, l]) for l in [0, 1]]) for m in [0, 1]]
        MINW_list[2:4] = [min([np.amin(W[:, :, l]) for l in [2, 3]]) for m in [0, 1]]
    elif RFI.UV.Npols == 2:
        MAXW_list = range(2)
        MAXW_list = [max([np.amaxW[:, :, l] for l in [0, 1]]) for m in [0, 1]]

        MINW_list = range(2)
        MINW_list = [min([np.amaxW[:, :, l] for l in [0, 1]]) for m in [0, 1]]
    else:
        MAXW_list = [np.amax(W[:, :, 0]), ]
        MINW_list = [np.amin(W[:, :, 0]), ]

    return(MAXW_list, MINW_list)


def grid_setup(RFI):

    if RFI.UV.Npols == 4:
        gs = GridSpec(3, 2)
        gs_loc = [[1, 0], [1, 1], [2, 0], [2, 1]]
    elif RFI.UV.Npols == 2:
        gs = GridSpec(2, 2)
        gs_loc = [[1, 0], [1, 1]]
    else:
        gs = GridSpec(2, 1)
        gs_loc = [[1, 0], ]

    return(gs, gs_loc)


def ax_constructor(RFI):

    if RFI.UV.Npols == 4:
        fig, ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    elif RFI.UV.Npols == 2:
        fig, ax = plt.subplots(figsize=(14, 8), nrows=2)
    else:
        fig, ax = plt.subplots(figsize=(14, 8))

    return(fig, ax)


def ax_chooser(RFI, ax, m):

    if RFI.UV.Npols == 4:
        curr_ax = ax[m / 2][m % 2]
    elif RFI.UV.Npols == 2:
        curr_ax = ax[m]
    else:
        curr_ax = ax

    return(curr_ax)


def polynomial_fit(x, y, order=1):

    basis = [x**(n) for n in range(order)]
    basis.append(np.ones(len(x)))

    A = np.concatenate(basis, axis=1)
    C, _, _, _ = scipy.linalg.lstsq(A, y)
    fit = np.sum(np.array([C[n] * x**(order - n) for n in range(order + 1)]))

    return(C, fit)


def planar_fit(x, y, z):

    A = np.c_[x, y, np.ones(len(x))]
    C, _, _, _ = scipy.linalg.lstsq(A, y)
    fit = C[0] * x + C[1] * y + C[2]

    return(C, fit)


def make_outdirs(RFI):

    flag_list = ['All', 'Post_Flag']
    spw_list = range(RFI.UV.Nspws)
    pairs = list(itertools.product(flag_list, spw_list))
    figpath = '%sfigs/' % (RFI.outpath)
    for pair in pairs:
        path = '%s%s/spw%i/' % (figpath, pair[0], pair[1])
        if not os.path.exists(path):
            os.makedirs(path)
    return(figpath)


def waterfall_catalog(RFI, amp_range={False: [1e3, 1e5], True: 'fit'},
                      fit={False: False, True: True}, bins=None,
                      fraction=True, bin_window=[0, 1e+03], aspect_ratio=3):

    """
    For each spectral window present in the data, construct a 1-d histogram,
    accompanied by a waterfall plot which reverse indexes elements of the
    data set within a chosen amplitude range. In each histogram plot is a
    histogram of all the data and a histogram of only data which passed through
    the flagger unmarked. The waterfall plots are separated by polarization, and
    show the total number of "affected" baselines at a given time/frequency in
    that polarization.
    """
    flag_labels = {True: 'Post-Flagging', False: 'All Baselines'}
    zorder = {'Post-Flagging': 3, 'Post-Flagging Fit': 4, 'All Baselines Fit': 2, 'All Baselines': 1}
    figpath = make_outdirs(RFI)
    for m in range(RFI.UV.Nspws):
        counts = []
        labels = []
        for flag in [True, False]:
            count, bins, hist_fit = RFI.one_d_hist_prepare(flag=flag,
                                                           fit=fit[flag],
                                                           bins=bins,
                                                           bin_window=bin_window)

            counts.append(count)
            labels.append(flag_labels[flag])
            if fit[flag]:
                counts.append(hist_fit)
                labels.append('%s Fit' % (flag_labels[flag]))

        # Set up a grid with gridspec
        gs, gs_loc = grid_setup(RFI)

        for flag in [True, False]:
            if amp_range[flag] is 'fit':
                amp_range[flag] = band_constructor(counts, bins, labels, flag_labels[flag])

            W = RFI.waterfall_hist_prepare(amp_range[flag], fraction=fraction,
                                           flag=flag)

            fig = plt.figure(figsize=(14, 8))
            ax = fig.add_subplot(gs[0, :])

            plot_lib.one_d_hist_plot(fig, ax, bins, counts, labels=labels,
                                     zorder=[zorder[label] for label in labels],
                                     xlabel='Amplitude (%s)' % (RFI.UV.vis_units),
                                     title='%s Visibility Difference Histogram' %
                                     (RFI.obs))

            ax.axvline(x=min(amp_range[flag]), color='black')
            ax.axvline(x=max(amp_range[flag]), color='black')

            MAXW_list, MINW_list = ext_list_selector(RFI, W[:, m, :, :])

            _, _, _, xticks, xminors, _, xticklabels = \
                plot_lib.four_panel_tf_setup(RFI.UV.freq_array[m, :])

            for n, pol in enumerate(RFI.pols):
                if fraction:
                    cbar_label = 'Fraction RFI'
                else:
                    cbar_label = 'Counts RFI'
                ax = fig.add_subplot(gs[gs_loc[n][0], gs_loc[n][1]])
                plot_lib.image_plot(fig, ax, W[:, m, :, n], cmap=cm.cool,
                                    vmin=MINW_list[n], vmax=MAXW_list[n],
                                    title='%s %s' % (pol, flag_labels[flag]),
                                    aspect_ratio=aspect_ratio, cbar_label=cbar_label,
                                    xticks=xticks, xminors=xminors, yminors='auto',
                                    xticklabels=xticklabels, mask_color='white')

            plt.tight_layout()
            fig.savefig('%s%s/spw%i/%s_%s_waterfall.png' % (figpath,
                                                            RFI.flag_titles[flag],
                                                            m, RFI.obs,
                                                            RFI.flag_titles[flag]))

            plt.close(fig)


def INS_catalog(RFI, aspect_ratio=3, invalid_mask=False, mask=True,
                sig_thresh=5):

    """
    Makes incoherent noise spectra for an obs. Requires a target directory.

    flag_slices: Compute the INS depending on which members of the visibility pairs were flagged
                 All, Flagged (boolean or), Unflagged (Neither seen as contaminated by COTTER)
    aspect_ratio: You will want to modify this if different than 80kHz
    invalid_mask: choose to mask nan's should they be present (they shouldn't be)
    mask: Choose to flag the outputs according to predictions made by the Central Limit Theorem
          This option is experimental right now and very incomplete
    """

    plot_titles = {False: 'All Baselines', True: 'Post-Flagging'}
    flag_titles = ['', 'Masked']
    figpath = make_outdirs(RFI)

    for flag in [True, False]:
        INS, MS, Nbls_arr, n, bins, fit = \
            RFI.INS_prepare(flag=flag)

        for p in range(1 + mask):
            if p > 0:
                INS, MS, n, bins, fit = \
                    rfiutil.match_filter(INS, MS, Nbls_arr, RFI.UV.freq_array,
                                         sig_thresh, RFI.obs)

            fig_hist, ax_hist = plt.subplots(figsize=(14, 8))
            plot_lib.one_d_hist_plot(fig_hist, ax_hist, bins, [n, fit],
                                     zorder=[2, 1], labels=['Data', 'Fit'],
                                     xlog=False, xlabel='Fraction of Mean',
                                     title='Incoherent Noise Spectrum Fractional Deviation Histogram %s' % (flag_titles[p]))

            for k in range(RFI.UV.Nspws):
                fig, ax = ax_constructor(RFI)
                _, _, _, xticks, xminors, yminors, xticklabels = plot_lib.four_panel_tf_setup(RFI.UV.freq_array[k, :])
                fig_diff, ax_diff = ax_constructor(RFI)

                fig.suptitle('%s Incoherent Noise Spectrum, %s %s' %
                             (RFI.obs, plot_titles[flag], flag_titles[p]))
                fig_diff.suptitle('%s Incoherent Noise Spectrum Fractional Deviation from Mean, %s %s' %
                                  (RFI.obs, plot_titles[flag], flag_titles[p]))

                for m, pol in enumerate(RFI.pols):
                    curr_ax = ax_chooser(RFI, ax, m)
                    curr_ax_diff = ax_chooser(RFI, ax_diff, m)

                    plot_lib.image_plot(fig, curr_ax, INS[:, k, :, m],
                                        title=pol, cbar_label=RFI.UV.vis_units,
                                        xticks=xticks, xminors=xminors,
                                        xticklabels=xticklabels, yminors=yminors,
                                        aspect_ratio=aspect_ratio, invalid_mask=invalid_mask,
                                        zero_mask=False)
                    plot_lib.image_plot(fig_diff, curr_ax_diff, MS[:, k, :, m],
                                        cmap=cm.coolwarm, title=pol,
                                        cbar_label='Fraction of Mean', xticks=xticks,
                                        xminors=xminors, xticklabels=xticklabels,
                                        yminors=yminors, aspect_ratio=aspect_ratio,
                                        invalid_mask=invalid_mask, zero_mask=False,
                                        mask_color='black')

                base = '%s%s/spw%i/%s_%s' % (figpath, RFI.flag_titles[flag], k,
                                             RFI.obs, flag_titles[p])
                fig.savefig('%s_INS.png' % (base))
                fig_diff.savefig('%s_INS_MS.png' % (base))
                fig_hist.savefig('%s_INS_hist.png' % (base))
                plt.close(fig)
                plt.close(fig_diff)
                plt.close(fig_hist)


def ant_scatter_catalog(RFI, outpath, band, flag_slice='All'):
    ant_locs = RFI.ant_scatter_prepare()
    H, uniques = RFI.drill_hist_prepare(band, flag_slice='All', drill_type='freq')

    for i in range(len(uniques)):
        for k in range(len(RFI.UV.Ntimes - 1)):
            fig, ax = ax_constructor(RFI)
            fig.suptitle('RFI Antenna Lightup t%i f%.1f' % (k, RFI.UV.freq_array[0, uniques[i]]))
            for m in range(len(RFI.UV.Npols)):
                c = np.array(RFI.UV.Nants_telescope * ['b'])
                c[H[:, k, m, uniques[i]] > 0] = 'r'
                curr_ax = ax_chooser(RFI, ax, m)
                plot_lib.scatter_plot_2d(fig, curr_ax, ant_locs[:, 0], ant_locs[:, 1],
                                         title=RFI.pols[m], xlabel='X (m)', ylabel='Y (m)',
                                         c=c)
            fig.savefig('%s%s_Ant_Scatter_f%i_t%i' % (outpath, RFI.obs, uniques[i], k))
            plt.close(fig)


def ant_pol_catalog(RFI, outpath, times=[], freqs=[], band=[], clip=False):

    if band:
        ind = RFI.reverse_index(band, flag_slice='All')
        times = ind[0]
        freqs = ind[3]

    for (time, freq) in zip(times, freqs):
        if not os.path.exists('%s%s_ant_pol_t%i_f%i.png' %
                              (outpath, RFI.obs, time, freq)):

            fig, ax = plt.subplots(figsize=(14, 8))
            T = RFI.ant_pol_prepare(time, freq, amp=clip)
            title = '%s Ant-Pol Drill t = %i f = %.1f Mhz ' % \
                    (RFI.obs, time, RFI.UV.freq_array[0, freq] * 10 ** (-6))
            vmax = np.amax(T)
            if clip:
                vmin = min(band)
            else:
                vmin = np.amin(T)

            plot_lib.image_plot(fig, ax, T, vmin=vmin, vmax=vmax, title=title,
                                aspect_ratio=1, xlabel='Antenna 2 Index',
                                ylabel='Antenna 1 Index',
                                cbar_label=RFI.UV.vis_units)

            fig.savefig('%s%s_ant_pol_t%i_f%i.png' % (outpath, RFI.obs,
                                                      time, freq))
            plt.close(fig)


def flag_catalog(RFI, outpath, flag_slices=['Flagged', ], xticks=None,
                 xminors=None, fraction=True):
    """
    Generate waterfall plots of flags, summed over baselines. In other words,
    how many baselines at a given time-pair/freq/pol were of a certain flag variety.
    Set fraction=True for the fraction of baselines.
    """
    for flag_slice in flag_slices:
        flags = RFI.flag_operations(flag_slice)
        flags = np.sum(flags, axis=1)
        if fraction:
            flags = flags.astype(float) / RFI.UV.Nbls

        fig, ax = ax_constructor(RFI)
        fig.suptitle('%s Visibility Difference Flag Map (%s)' %
                     (RFI.obs, flag_slice))

        for m in range(RFI.UV.Npols):
            curr_ax = ax_chooser(RFI, ax, m)

            plot_lib.image_plot(fig, curr_ax, flags[:, 0, :, m], cmap=cm.cool,
                                title=RFI.pols[m],
                                xticks=xticks, xminors=xminors,
                                xticklabels=['%.1f' % ((10 ** (-6)) *
                                             RFI.UV.freq_array[0, tick]) for
                                             tick in xticks])

        fig.savefig('%s%s_flag_map_%s.png' % (outpath, RFI.obs, flag_slice))


def bl_scatter_catalog(RFI, cmap=cm.plasma, gridsize=50, flag=False, sig_thresh=4,
                       shape_dict={}, edges=np.linspace(-3000, 3000, num=51)):

    INS, MS, Nbls, _, _, _ = RFI.INS_prepare(flag=flag, sig_thresh=sig_thresh)
    INS, MS, n, bins, fit, events = rfiutil.match_filter(INS, MS, Nbls,
                                                         RFI.UV.freq_array,
                                                         sig_thresh, shape_dict,
                                                         '%sarrs/' % (RFI.outpath))
    if not os.path.exists('%sfigs/' % (RFI.outpath)):
        os.makedirs('%sfigs/' % (RFI.outpath))
    if len(events) > 0:
        grid, bl_hist, bl_bins, sim_hist, cutoffs = \
            RFI.bl_grid_flag(events, gridsize=gridsize, edges=edges, flag=flag)

        xticklabels = ['%.0f' % (edges[tick]) for tick in range(0, 50, 10)]
        yticklabels = ['%.0f' % (edges[tick]) for tick in range(50, 0, -10)]

        for m in range(grid.shape[2]):

            fig_hist, ax_hist = plt.subplots(figsize=(14, 8))
            fig_grid, ax_grid = plt.subplots(figsize=(14, 8))

            title_tuple = (RFI.obs,
                           min(RFI.UV.freq_array[events[m, 0], events[m, 2]]) * 10 ** (-6),
                           max(RFI.UV.freq_array[events[m, 0], events[m, 2]]) * 10 ** (-6),
                           events[m, 3].indices(RFI.UV.Ntimes - 1)[0],
                           events[m, 3].indices(RFI.UV.Ntimes - 1)[1],
                           RFI.pols[events[m, 1]])

            plot_lib.one_d_hist_plot(fig_hist, ax_hist, bl_bins[m], [bl_hist[m], sim_hist[m]], xlog=False,
                                     labels=['Measurements', 'Monte Carlo'], xlabel='Amplitude (Median)',
                                     title='%s RFI Event-Averaged Amplitude Histogram %.1f - %.1f Mhz, t%i - t%i, %s' % title_tuple)
            ax_hist.axvline(x=cutoffs[m], color='black')
            plot_lib.image_plot(fig_grid, ax_grid, grid[:, :, m],
                                title='%s RFI Baseline Gridded Average, %.2f - %.2f Mhz, t%i - t%i, %s' % title_tuple,
                                aspect_ratio=1, xlabel='$\lambda u$ (m)',
                                ylabel='$\lambda v$ (m)', xticklabels=xticklabels,
                                yticklabels=yticklabels,
                                cbar_label='Amp (%s)' % (RFI.UV.vis_units))

            fig_hist.savefig('%sfigs/%s_event_hist_%i.png' % (RFI.outpath, RFI.obs, m))
            fig_grid.savefig('%sfigs/%s_event_bl_grid_%i.png' % (RFI.outpath, RFI.obs, m))

            plt.close(fig_hist)
            plt.close(fig_grid)
    else:
        print('No events were found in the incoherent noise spectra!')


def one_d_hist_catalog(RFI, norm=False, pow=False, MC=False):
    fig, ax = plt.subplots(figsize=(14, 8), nrows=2)
    fig.suptitle('%s Visibility Difference Amplitude Histogram' % (RFI.obs))
    amp_labels = ('Amplitude (UNCALIB)', 'Amplitude (Sigma)')
    for i, norm in enumerate([False, True]):
        fit = not norm
        n, bins, fit = RFI.one_d_hist_prepare(flag=True, bins=None, fit=fit,
                                              norm=norm, MC=MC, pow=pow)
        plot_lib.one_d_hist_plot(fig, ax[i], bins, [n, fit], labels=['Data', 'Fit'],
                                 xlog=not norm, xlabel=amp_labels[i])
    if not os.path.exists('%sfigs/' % (RFI.outpath)):
        os.makedirs('%sfigs/' % (RFI.outpath))
    fig.savefig('%sfigs/%s_Amp_Hist.png' % (RFI.outpath, RFI.obs))
