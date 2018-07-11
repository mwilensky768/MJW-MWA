from matplotlib import cm, use
use('Agg')
import matplotlib.pyplot as plt
import plot_lib as pl


def fig_construct(N_ax):

    fig_kwargs = {'figsize': (14, 8),
                  'nrows': N_ax / (2 - N_ax % 2),
                  'ncols': 2 - N_ax % 2}

    return(plt.subplots(**fig_kwargs))


def INS(RFI, data, outpath, plot_kwargs={}, data_kwargs={}):

    im_kwargs = [{'cmap': cm.plasma,
                  'cbar_label': 'Amplitude (%s)' % (RFI.UV.vis_units)},
                 {'cmap': cm.coolwarm,
                  'cbar_label': 'Fraction of Mean',
                  'mask_color': 'black'}]

    im_titles = ['%s Incoherent Noise Spectrum' % (RFI.obs),
                 '%s Mean-Subtracted Incoherent Noise Spectrum' % (RFI.obs)]
    im_tags = ['INS',
               'MS']

    for i in range(data[0].shape[1]):
        fig_INS, ax_INS = fig_construct(data[0].shape[3])
        fig_MS, ax_MS = fig_construct(data[0].shape[3])

        for k, (fig, ax) in enumerate(zip([fig_INS, fig_MS], [ax_INS, ax_MS])):
            plot_kwargs.update(im_kwargs[k])
            for m in range(data[0].shape[3]):
                fig.suptitle(im_titles[k])
                im_args = [fig, ax[m / 2][m % 2], data[k][:, i, :, m]]
                pl.image_plot(*im_args, title=RFI.pols[m], **plot_kwargs)
            fig.savefig('%s/%s_%s_spw%i_%s.png' % (outpath, RFI.obs,
                                                   RFI.flag_titles[data_kwargs['choice']],
                                                   i, im_tags[k]))
            plt.close(fig)


def bl_grid(RFI, data, outpath, plot_kwargs={}):

    im_kwargs = {'xlabel': None,
                 'ylabel': None,
                 'cbar_label': 'Amplitude (%s)' % (RFI.UV.vis_units)}
    plot_kwargs.update(im_kwargs)

    hist_kwargs = {'labels': ['Measurements', 'Monte Carlo'],
                   'xlog': False,
                   'xlabel': 'Amplitude (%s)' % (RFI.UV.vis_units)}

    for i in data[0].shape[2]:
        fig_grid, ax_grid = fig_construct(1)
        fig_hist, ax_hist = fig_construct(1)

        title_tuple = (RFI.obs,
                       data[5][i, 3],
                       min(RFI.UV.freq_array[data[5][i, 0], data[5][i, 2]]) * 10 ** (-6),
                       max(RFI.UV.freq_array[data[5][i, 0], data[5][i, 2]]) * 10 ** (-6),
                       RFI.pols[data[5][i, 1]])

        title = '%s, t%i, %.2f Mhz - %.2f Mhz, %s, ' % (title_tuple)
        subtitles = ['UV Grid', 'Event Histogram']
        args = [(data[0][:, :, i], ), (data[1][i], [data[2][i], data[3][i]])]
        kwargs = [plot_kwargs, hist_kwargs]
        plot_type = ['image_plot, one_d_hist_plot']
        tags = ['bl_grid', 'hist']

        for k, (fig, ax) in enumerate(zip([fig_grid, fig_hist], [ax_grid, ax_hist])):
            getattr(pl, plot_type[k])(fig, ax, *args[k], title=title + subtitles[k], **kwargs[k])
            fig.savefig('%sfigs/bl_grid/%s_t%i_f%.2f_f%.2f_%s_%s.png' %
                        (RFI.outpath, ) + title_tuple + (tags[k], ))
            plt.close(fig)
