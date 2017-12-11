import matplotlib.pyplot as plt
from matplotlib import cm, use
use('Agg')
from matplotlib.ticker import FixedLocator, AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def one_d_hist_plot(fig, ax, bin_edges, counts, zorder=[], labels=[], xlog=True,
                    ylog=True, xlabel='Amplitude', ylabel='Counts', title=''):

    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + 0.5 * bin_widths

    for i in range(len(labels)):
        ax.step(bin_centers, counts[i], where='mid', label=labels[i],
                zorder=zorder[i])

    ax.set_ylim([10**(-1), 10 * max([np.amax(x) for x in counts])])
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if xlog:
        ax.set_xscale('log', nonposy='clip')
    else:
        ax.set_xscale('linear')

    if ylog:
        ax.set_yscale('log', nonposy='clip')
    else:
        ax.set_yscale('linear')


def line_plot(fig, ax, data, title='Visibility Difference Average',
              xlabel='Frequency (Mhz)', ylabel='Visibility Amplitude',
              zorder=[], labels=[], xticks=[], xticklabels=[], xminors=[]):  # Please pass data as a list

    for k in range(len(data)):
        ax.plot(data[k], label=labels[k], zorder=zorder[k])

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
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
               mask_color='white'):

    if zero_mask:
        data = np.ma.masked_equal(data, 0)

    cmap = cmap
    cmap.set_bad(color=mask_color)

    if vmin is None:
        vmin = np.amin(data)
    if vmax is None:
        vmax = np.amax(data)

    cax = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
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
                    c=[]):

    if c:
        ax.scatter(x, y, c=c)
    else:
        ax.scatter(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def scatter_plot_3d(fig, ax, x_data, y_data, z_data, title='', xlabel='',
                    ylabel='', zlabel='', c=[]):

    if c:
        ax.scatter(x, y, z, c=c)
    else:
        ax.scatter(x, y, z)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
