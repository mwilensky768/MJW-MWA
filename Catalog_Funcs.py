import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import plot_lib


def band_constructor(counts, bins, fit):

    max_loc = min(bins[counts == np.amax(counts)])
    band = [min(bins[:-1][np.logical_and(fit < 1, bins[:-1] > max_loc)]),
            10 * max(bins)]

    return(band)


def ext_list_selector(RFI, W):

    if RFI.UV.Npols > 1:
        MAXW_list = range(4)
        MAXW_list[:2] = [max([np.amax(W[:, :, l]) for l in [0, 1]]) for m in [0, 1]]
        MAXW_list[2:4] = [max([np.amax(W[:, :, l]) for l in [2, 3]]) for m in [0, 1]]

        MINW_list = range(4)
        MINW_list[:2] = [min([np.amin(W[:, :, l]) for l in [0, 1]]) for m in [0, 1]]
        MINW_list[2:4] = [min([np.amin(W[:, :, l]) for l in [2, 3]]) for m in [0, 1]]
    else:
        MAXW_list = [np.amax(W[:, 0, :, :]), ]
        MINW_list = [np.amin(W[:, 0, :, :]), ]

    return(MAXW_list, MINW_list)


def grid_setup(RFI):

    if RFI.UV.Npols > 1:
        gs = GridSpec(3, 2)
        gs_loc = [[1, 0], [1, 1], [2, 0], [2, 1]]
    else:
        gs = GridSpec(2, 1)
        gs_loc = [[1, 0], ]

    return(gs, gs_loc)


def waterfall_catalog(RFI, outpath, band={}, write={}, writepath='', fit={},
                      bins=np.logspace(-3, 5, num=1001), fraction=True,
                      flag_slices=['Unflagged', 'All'], bin_window=[0, 1e+03],
                      zorder={'Unflagged': 4, 'Unflagged Fit': 3, 'All Fit': 2, 'All': 1},
                      xticks=None, xminors=None):

    counts = []
    labels = []
    for flag_slice in flag_slices:
        count, _, hist_fit, label = RFI.one_d_hist_prepare(flag_slice=flag_slice,
                                                           fit=fit[flag_slice],
                                                           write=write[flag_slice],
                                                           bins=bins,
                                                           writepath=writepath,
                                                           bin_window=bin_window,
                                                           label=flag_slice)

        counts.append(count)
        counts.append(hist_fit)
        labels.append(label)
        labels.append('%s Fit' % (label))

    labels = np.array(labels)[[item is not None for item in counts]]
    counts = np.array(counts)[[item is not None for item in counts]]
    gs, gs_loc = grid_setup(RFI)

    for flag_slice in flag_slices:
        if band[flag_slice] is 'fit':
            band[flag_slice] = band_constructor(counts[flag_slice], bins,
                                                hist_fit[flag_slice])

        W = RFI.waterfall_hist_prepare(band[flag_slice], fraction=fraction,
                                       flag_slice=flag_slice)

        fig = plt.figure(figsize=(14, 8))
        ax = fig.add_subplot(gs[0, :])

        plot_lib.one_d_hist_plot(fig, ax, bins, counts, labels=labels,
                                 zorder=[zorder[label] for label in labels],
                                 xlabel='Amplitude (%s)' % (RFI.UV.vis_units),
                                 title='%s Visibility Difference Histogram' %
                                 (RFI.obs))

        ax.axvline(x=min(band[flag_slice]), color='black')
        ax.axvline(x=max(band[flag_slice]), color='black')

        MAXW_list, MINW_list = ext_list_selector(RFI, W[:, 0, :, :])

        for n in range(self.UV.Npols):
            ax = fig.add_subplot(gs[gs_loc[n][0], gs_loc[n][1]])
            plot_lib.image_plot(fig, ax, W[:, 0, :, n], cmap=cm.coolwarm,
                                vmin=MINW_list[n], vmax=MAXW_list[n],
                                title='%s %s' % (RFI.pols[n], flag_slice),
                                aspect_ratio=3, cbar_label='Fraction RFI',
                                xticks=xticks, xminors=xminors, yminors='auto',
                                xticklabels=['%.1f' % (RFI.UV.freq_array[0, tick])
                                             for tick in xticks])

        plt.tight_layout()
        fig.savefig('%s%s_freq_time_%s.png' % (outpath, RFI.obs, flag_slice))

        plt.close(fig)


def drill_catalog(RFI, outpath, band={}, write={}, writepath='', fit={},
                  bins=np.logspace(-3, 5, num=1001),
                  flag_slices=['Unflagged', 'All'], bin_window=[0, 1e+03],
                  zorder={'Unflagged': 4, 'Unflagged Fit': 3, 'All Fit': 2, 'All': 1},
                  xticks=None, xminors=None, drill_type='time'):

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

            MAXW_list, MINW_list = ext_list_selector(RFI, W[:, :, :, uniques[k]])

            for n in range(self.UV.Npols):
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


def vis_avg_catalog(RFI, outpath, band=[1.5 * 10**3, 10**5], flag_slice='All',
                    bl_slice='All', amp_avg='Amp', plot_type='waterfall',
                    xticks=[], xminors=[], yticks=[], yminors=[]):

    data = RFI.vis_avg_prepare(band=band, flag_slice=flag_slice,
                               bl_slice=bl_slice, amp_avg=amp_avg)

    if plot_type is 'waterfall':
        if RFI.UV.Npols > 1:
            fig, ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
        else:
            fig, ax = plt.subplots(figsize=(14, 8))
        fig.suptitle('%s Visibility Difference Averages, %s First' %
                     (RFI.obs, amp_avg))
        for m in range(RFI.UV.Npols):
            if RFI.UV.Npols > 1:
                curr_ax = ax[m / 2][m % 2]
            else:
                curr_ax = ax
            plot_lib.image_plot(fig, curr_ax, data[:, 0, :, m],
                                title='%s %s Flags %s Bls' %
                                (RFI.pols[m], flag_slice, bl_slice),
                                cbar_label='%s' % (RFI.UV.vis_units),
                                xticks=xticks, xminors=xminors,
                                xticklabels=['%.1f' %
                                             (10 ** (-6) * RFI.UV.freq_array[0, int(tick)])
                                             for tick in xticks],
                                yticks=yticks, yminors=yminors)
        fig.savefig('%s%s_Vis_Avg_Waterfall.png' % (outpath, RFI.obs))
        plt.close(fig)
    else:
        for m in range(RFI.UV.Ntimes - 1):
            if RFI.UV.Npols > 1:
                fig, ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
            else:
                fig, ax = plt.subplots(figsize=(14, 8))
            for n in range(RFI.UV.Npols):
                if RFI.UV.Npols > 1:
                    curr_ax = ax[n / 2][n % 2]
                else:
                    curr_ax = ax
                plot_lib.line_plot(fig, curr_ax, ylabel=RFI.UV.vis_units,
                                   labels=['Affected Baselines', 'Unaffected Baselines'],
                                   zorder=[1, 2], xticks=xticks,
                                   xticklabels=['%.1f' % (10 ** (-6) *
                                                          RFI.UV.freq_array[0, 0] +
                                                          RFI.UV.channel_width *
                                                          tick) for ticks in xticks],
                                   data=[subdata[m, 0, :, n] for subdata in data],
                                   title='%s %s Flags' % (RFI.pols[n], flag_slice))
            fig.suptitle('%s Visibility Difference Average (%s First)' %
                         (RFI.obs, amp_avg))
            fig.savefig('%s%s_Vis_Avg_t%i.png' % (outpath, RFI.obs, m))
            plt.close(fig)