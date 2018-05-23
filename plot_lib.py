from matplotlib import cm, use
use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.colors as colors


class MidpointNormalize(colors.Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        result, is_scalar = self.process_value(value)
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.array(np.interp(value, x, y), mask=result.mask, copy=False)


def four_panel_tf_setup(freq_array):
    fig, ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    pols = ['XX', 'YY', 'XY', 'YX']
    Nfreqs = len(freq_array)
    xticks = [Nfreqs * k / 6 for k in range(6)]
    xticks.append(Nfreqs - 1)
    xminors = AutoMinorLocator(4)
    yminors = 'auto'
    xticklabels = ['%.1f' % (freq_array[tick] * 10 ** (-6)) for tick in xticks]

    return(fig, ax, pols, xticks, xminors, yminors, xticklabels)


def one_d_hist_plot(fig, ax, bin_edges, counts, zorder=None, labels=None, xlog=True,
                    ylog=True, xlabel='Amplitude', ylabel='Counts', title='',
                    legend=True):
    # Must give counts in a list

    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + 0.5 * bin_widths

    if not zorder:
        zorder = range(len(counts))
    if not labels:
        labels = len(counts) * ['']

    for i, count in enumerate(counts):
        ax.step(bin_centers, count, where='mid', label=labels[i],
                zorder=zorder[i])

    ax.set_ylim([10**(-1), 10 * max([np.amax(x) for x in counts])])
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if legend:
        ax.legend()

    if xlog:
        ax.set_xscale('log', nonposx='clip')
    else:
        ax.set_xscale('linear')

    if ylog:
        ax.set_yscale('log', nonposy='clip')
    else:
        ax.set_yscale('linear')


def line_plot(fig, ax, data, title='Visibility Difference Average',
              xlabel='Frequency (Mhz)', ylabel='Visibility Amplitude',
              zorder=None, labels=None, xticks=None, xticklabels=None,
              xminors=None, legend=True):  # Please pass data as a list

    if not zorder:
        zorder = range(len(data))
    if not labels:
        labels = len(data) * ['']

    for k in range(len(data)):
        ax.plot(data[k], label=labels[k], zorder=zorder[k])

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    if legend:
        ax.legend()
    if xticks:
        ax.set_xticks(xticks)
    if xticklabels:
        ax.set_xticklabels(xticklabels)
    if xminors:
        ax.xaxis.set_minor_locator(xminors)


def image_plot(fig, ax, data, cmap=cm.plasma, vmin=None, vmax=None, title='',
               aspect_ratio=3, xlabel='Frequency (Mhz)', ylabel='Time Pair',
               cbar_label='Counts RFI', xticks=[], yticks=[], xminors=None,
               yminors=None, xticklabels=None, yticklabels=None, zero_mask=True,
               mask_color='white', invalid_mask=False):

    if zero_mask:
        data = np.ma.masked_equal(data, 0)
    if invalid_mask:
        data = np.ma.masked_invalid(data)

    cmap = cmap

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)

    if cmap is cm.coolwarm:
        cax = ax.imshow(data, cmap=cmap, clim=(vmin, vmax),
                        norm=MidpointNormalize(midpoint=0, vmin=vmin, vmax=vmax))
    else:
        cax = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)

    cmap.set_bad(color=mask_color)
    cbar = fig.colorbar(cax, ax=ax)
    cbar.set_label(cbar_label)

    ax.set_title(title)

    if xticks:
        ax.set_xticks(xticks)
    if yticks:
        ax.set_yticks(yticks)
    if xminors is 'auto':
        xminors = AutoMinorLocator(ax.get_xticks()[1] - ax.get_xticks()[0])
        ax.xaxis.set_minor_locator(xminors)
    elif xminors:
        ax.xaxis.set_minor_locator(xminors)
    if yminors is 'auto':
        yminors = AutoMinorLocator(ax.get_yticks()[1] - ax.get_yticks()[0])
        ax.yaxis.set_minor_locator(yminors)
    elif yminors:
        ax.yaxis.set_minor_locator(yminors)
    if xticklabels:
        ax.set_xticklabels(xticklabels)
    if yticklabels:
        ax.set_yticklabels(yticklabels)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_aspect(aspect_ratio)


def scatter_plot_2d(fig, ax, x_data, y_data, title='', xlabel='', ylabel='',
                    c=None, ylim=None, cmap=None, vmin=None, vmax=None, norm=None,
                    cbar_label=None, s=None, xticks=None, yticks=None):

    cax = ax.scatter(x_data, y_data, c=c, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm,
                     s=s)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    if cmap is not None:
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label(cbar_label)
    if ylim:
        ax.set_ylim(ylim)


def scatter_plot_3d(fig, ax, x_data, y_data, z_data, title='', xlabel='',
                    ylabel='', zlabel='', c=None, cmap=None, vmin=None,
                    vmax=None):

    ax.scatter(x, y, z, c=c, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
