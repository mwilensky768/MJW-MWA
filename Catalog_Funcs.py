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


def band_constructor(counts, bins, labels, flag_slice):

    for k, label in enumerate(labels):
        if label == flag_slice:
            m = k

    count = counts[m]
    fit = counts[m + 1]

    max_loc = min(bins[:-1][count.argmax()])
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


def make_outdirs(outpath, flag_slice, spw_sep=True, m=None):

    if spw_sep:
        writepath = '%s%s/spw%i/arrs/' % (outpath, flag_slice, m)
        figpath = '%s%s/spw%i/figs/' % (outpath, flag_slice, m)
    else:
        writepath = '%s%s/all_spw/arrs/' % (outpath, flag_slice)
        figpath = '%s%s/all_spw/figs/' % (outpath, flag_slice)
    if not os.path.exists(writepath):
        os.makedirs(writepath)
    if not os.path.exists(figpath):
        os.makedirs(figpath)

    return(writepath, figpath)


def waterfall_catalog(RFI, outpath, band={}, fit_type={},
                      bins=np.logspace(-3, 5, num=1001), fraction=True,
                      flag_slices=['Unflagged', 'All'], bin_window=[0, 1e+03],
                      aspect_ratio=3):

    """
    """

    zorder = {'Unflagged': 3, 'Unflagged Fit': 4, 'All Fit': 2, 'All': 1}
    for m in range(RFI.UV.Nspws):
        counts = []
        labels = []
        for flag_slice in flag_slices:
            writepath, figpath = make_outdirs(outpath, flag_slice, m=m)
            count, bins, hist_fit = RFI.one_d_hist_prepare(flag_slice=flag_slice,
                                                           fit=fit_type[flag_slice],
                                                           bins=bins,
                                                           writepath=writepath,
                                                           bin_window=bin_window,
                                                           spw_ind=m)

            counts.append(count)
            labels.append(flag_slice)
            if fit_type[flag_slice]:
                counts.append(hist_fit)
                labels.append('%s Fit' % (flag_slice))

        gs, gs_loc = grid_setup(RFI)

        for flag_slice in flag_slices:
            if band[flag_slice] is 'fit':
                band[flag_slice] = band_constructor(counts, bins, labels, flag_slice)

            W = RFI.waterfall_hist_prepare(band[flag_slice], fraction=fraction,
                                           flag_slice=flag_slice, writepath=writepath)

            fig = plt.figure(figsize=(14, 8))
            ax = fig.add_subplot(gs[0, :])

            plot_lib.one_d_hist_plot(fig, ax, bins, counts, labels=labels,
                                     zorder=[zorder[label] for label in labels],
                                     xlabel='Amplitude (%s)' % (RFI.UV.vis_units),
                                     title='%s Visibility Difference Histogram' %
                                     (RFI.obs))

            ax.axvline(x=min(band[flag_slice]), color='black')
            ax.axvline(x=max(band[flag_slice]), color='black')

            MAXW_list, MINW_list = ext_list_selector(RFI, W[:, m, :, :])
            _, _, _, xticks, xminors, _, xticklabels = plot_lib.four_panel_tf_setup(RFI.UV.freq_array[m, :])

            for n, pol in enumerate(RFI.pols):
                if fraction:
                    cbar_label = 'Fraction RFI'
                else:
                    cbar_label = 'Counts RFI'
                ax = fig.add_subplot(gs[gs_loc[n][0], gs_loc[n][1]])
                plot_lib.image_plot(fig, ax, W[:, m, :, n], cmap=cm.cool,
                                    vmin=MINW_list[n], vmax=MAXW_list[n],
                                    title='%s %s' % (pol, flag_slice),
                                    aspect_ratio=aspect_ratio, cbar_label=cbar_label,
                                    xticks=xticks, xminors=xminors, yminors='auto',
                                    xticklabels=xticklabels, mask_color='white')

            plt.tight_layout()
            fig.savefig('%s%s_waterfall_%s.png' % (figpath, RFI.obs, flag_slice))
            print('I saved the fig')
            plt.close(fig)


def drill_catalog(RFI, outpath, band={}, write={}, writepath='', fit={},
                  bins=np.logspace(-3, 5, num=1001),
                  flag_slices=['Unflagged', 'All'], bin_window=[0, 1e+03],
                  zorder={'Unflagged': 4, 'Unflagged Fit': 3, 'All Fit': 2, 'All': 1},
                  xticks=None, xminors=None, drill_type='time'):

    """
    Do not use this function right now.
    """

    gs, gs_loc = grid_setup(RFI)

    for flag_slice in flag_slices:
        H, uniques = RFI.drill_hist_prepare(band[flag_slice],
                                            flag_slice=flag_slice,
                                            drill_type=drill_type)

        for k in range(len(uniques)):
            fig = plt.figure(figsize=(14, 8))
            ax = fig.add_subplot(gs[0, :])

            counts = []
            labels = []
            for flag in flag_slices:

                if drill_type is 'time':

                    count, _, hist_fit, label = \
                        RFI.one_d_hist_prepare(flag_slice=flag,
                                               time_drill=uniques[k],
                                               fit=fit[flag_slice], bins=bins,
                                               bin_window=bin_window,
                                               label='%s t = %i' %
                                               (flag_slice, uniques[k]))

                    count_exc, _, hist_fit_exc, label_exc = \
                        RFI.one_d_hist_prepare(flag_slice=flag,
                                               time_exc=uniques[k],
                                               fit=fit[flag_slice], bins=bins,
                                               bin_window=bin_window,
                                               label='%s t != %i' %
                                               (flag_slice, uniques[k]))

                    for item in [count, count_exc, hist_fit, hist_fit_exc]:
                        counts.append(item)
                    for item in [label, label_exc, '%s Fit' % (label),
                                 '%s Fit' % (label_exc)]:
                        labels.append(item)

                elif drill_type is 'freq':

                    count, _, hist_fit, label = \
                        RFI.one_d_hist_prepare(flag_slice=flag,
                                               freq_drill=uniques[k],
                                               fit=fit[flag_slice], bins=bins,
                                               bin_window=bin_window,
                                               label='%s f = %.1f Mhz' %
                                               (flag_slice, (10**(-6)) *
                                                RFI.UV.freq_array(0, uniques[k])))

                    count_exc, _, hist_fit_exc, label_exc = \
                        RFI.one_d_hist_prepare(flag_slice=flag,
                                               freq_exc=uniques[k],
                                               fit=fit[flag_slice], bins=bins,
                                               bin_window=bin_window,
                                               label='%s f != %.1f Mhz' %
                                               (flag_slice, (10 ** (-6)) *
                                                RFI.UV.freq_array(0, uniques[k])))

                    for item in [count, count_exc, hist_fit, hist_fit_exc]:
                        counts.append(item)
                    for item in [label, label_exc, '%s Fit' % (label),
                                 '%s Fit' % (label_exc)]:
                        labels.append(item)

            labels = np.array(labels)[[item is not None for item in counts]]
            counts = np.array(counts)[[item is not None for item in counts]]

            plot_lib.one_d_hist_plot(fig, ax, bins, counts, labels=labels,
                                     zorder=[zorder[label] for label in labels],
                                     xlabel='Amplitude (%s)' % (RFI.UV.vis_units),
                                     title='%s Visibility Difference Histogram' %
                                     (RFI.obs))

            ax.axvline(x=min(band[flag_slice]), color='black')
            ax.axvline(x=max(band[flag_slice]), color='black')

            MAXW_list, MINW_list = ext_list_selector(RFI, H[:, :, :, uniques[k]])

            for n in range(RFI.UV.Npols):
                ax = fig.add_subplot(gs[gs_loc[n][0], gs_loc[n][1]])
                if drill_type is 'time':
                    xticklabels = ['%.1f' % (RFI.UV.freq_array[0, tick])
                                   for tick in xticks]
                else:
                    xticklabels = []

                plot_lib.image_plot(fig, ax, H[:, :, n, uniques[k]], cmap=cm.coolwarm,
                                    vmin=MINW_list[n], vmax=MAXW_list[n],
                                    title='%s %s' % (RFI.pols[n], flag_slice),
                                    aspect_ratio=3, cbar_label='Counts RFI',
                                    xticks=xticks, xminors=xminors, yminors='auto',
                                    xticklabels=xticklabels)

            plt.tight_layout()
            fig.savefig('%s%s_%s_drill_%s_%s%i.png' % (outpath, RFI.obs, drill_type,
                                                       flag_slice, drill_type[0],
                                                       uniques[k]))

            plt.close(fig)


def INS_catalog(RFI, outpath, flag_slices=['All', ], amp_avg='Amp', aspect_ratio=3,
                invalid_mask=False, mask=True):

    """
    Makes incoherent noise spectra for an obs. Requires a target directory.

    flag_slices: Compute the INS depending on which members of the visibility pairs were flagged
                 All, Flagged (boolean or), Unflagged (Neither seen as contaminated by COTTER)
    amp_avg: 'Amp' = average the amplitudes of the differences
             'Avg' = average the differences, then take the amplitude
    aspect_ratio: You will want to modify this if different than 80kHz
    invalid_mask: choose to mask nan's should they be present (they shouldn't be)
    mask: Choose to flag the outputs according to predictions made by the Central Limit Theorem
          This option is experimental right now and very incomplete
    """

    plot_titles = {'All': 'All Baselines', 'Unflagged': 'Post-Flagging'}
    flag_titles = ['', 'Masked']

    for flag_slice in flag_slices:

        writepath, figpath = make_outdirs(outpath, flag_slice, spw_sep=False)

        INS, frac_diff, n, bins, fit = \
            RFI.INS_prepare(flag_slice=flag_slice, amp_avg=amp_avg,
                            writepath=writepath)

        for p in range(1 + mask):
            if p > 0:
                INS, frac_diff, n, bins, fit = \
                    rfiutil.INS_outlier_flag(RFI.obs, INS, frac_diff, RFI.UV.Nbls,
                                             flag_slice=flag_slice, amp_avg=amp_avg,
                                             writepath=writepath)

            fig_hist, ax_hist = plt.subplots(figsize=(14, 8))
            plot_lib.one_d_hist_plot(fig_hist, ax_hist, bins, [n, fit], zorder=[2, 1],
                                     labels=['Data', 'Fit'], xlog=False, xlabel='Fraction',
                                     title='Incoherent Noise Spectrum Fractional Deviation Histogram %s' % (flag_titles[p]))

            for k in range(RFI.UV.Nspws):
                fig, ax = ax_constructor(RFI)
                _, _, _, xticks, xminors, yminors, xticklabels = plot_lib.four_panel_tf_setup(RFI.UV.freq_array[k, :])
                fig_diff, ax_diff = ax_constructor(RFI)

                fig.suptitle('%s Incoherent Noise Spectrum, %s %s' %
                             (RFI.obs, plot_titles[flag_slice], flag_titles[p]))
                fig_diff.suptitle('%s Incoherent Noise Spectrum Fractional Deviation from Mean, %s %s' %
                                  (RFI.obs, plot_titles[flag_slice], flag_titles[p]))

                for m, pol in enumerate(RFI.pols):
                    curr_ax = ax_chooser(RFI, ax, m)
                    curr_ax_diff = ax_chooser(RFI, ax_diff, m)
                    plot_lib.image_plot(fig, curr_ax, INS[:, k, :, m],
                                        title=pol, cbar_label=RFI.UV.vis_units,
                                        xticks=xticks, xminors=xminors,
                                        xticklabels=xticklabels, yminors=yminors,
                                        aspect_ratio=aspect_ratio, invalid_mask=invalid_mask,
                                        zero_mask=False)
                    plot_lib.image_plot(fig_diff, curr_ax_diff, frac_diff[:, k, :, m],
                                        cmap=cm.coolwarm, title=pol,
                                        cbar_label='Fraction', xticks=xticks,
                                        xminors=xminors, xticklabels=xticklabels,
                                        yminors=yminors, aspect_ratio=aspect_ratio,
                                        invalid_mask=invalid_mask, zero_mask=False)
                base = '%s%s_spw%i_%s_%s' % (figpath, RFI.obs, k, flag_slice, flag_titles[p])
                print(INS.shape)
                fig.savefig('%s_INS.png' % (base))
                fig_diff.savefig('%s_INS_frac_diff.png' % (base))
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


def bl_scatter_catalog(RFI, outpath, mask, vmin=None, vmax=None, cmap=cm.plasma):
    """
    You must have at least numpy 1.13 to run this!
    """
    ind = np.where(mask)
    ind_arr = np.unique(np.vstack((ind[0:2])), axis=0)

    for k in range(len(ind[0])):
        fig, ax = ax_constructor(RFI)
        fig.suptitle('Baseline Scatter Plot t = %i, f = %.1f Mhz' %
                     (ind[0][k], RFI.UV.freq_array[ind[1][k], ind[2][k]] * 10**(-6)))
        figpath = '%s/All/spw%i/figs/' % (outpath, ind[1][k])
        if not os.path.exists:
            os.makedirs(figpath)
        for m, pol in enumerate(RFI.pols):
            curr_ax = ax_chooser(RFI, ax, m)
            if vmin is None:
                vmin = np.absolute(RFI.data_array[ind[0][k], :, ind[1][k], ind[2][k], m])[np.absolute(RFI.data_array[ind[0][k], :, ind[1][k], ind[2][k], m]) > 0].min()
            if vmax is None:
                vmax = np.absolute(RFI.data_array[ind[0][k], :, ind[1][k], ind[2][k], m]).max()
            plot_lib.scatter_plot_2d(fig, curr_ax,
                                     RFI.UV.uvw_array[ind[0][k] * RFI.UV.Nbls:ind[0][k] * (RFI.UV.Nbls + 1), 0],
                                     RFI.UV.uvw_array[ind[0][k] * RFI.UV.Nbls:ind[0][k] * (RFI.UV.Nbls + 1), 1],
                                     title=pol, xlabel='$\lambda u$ (m)', ylabel='$\lambda v$ (m)',
                                     c=np.absolute(RFI.data_array[ind[0][k], :, ind[1][k], ind[2][k], m]),
                                     cmap=cmap, vmin=vmin, vmax=vmax,
                                     norm=matplotlib.colors(LogNorm(vmin=vmin, vmax=vmax)),
                                     cbar_label=RFI.UV.vis_units)

        fig.savefig('%s%s_bl_scat_t%i_f%i.png' % (figpath, RFI.obs, ind[0][k], ind[2][k]))
