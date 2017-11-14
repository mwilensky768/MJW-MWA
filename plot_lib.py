import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FixedLocator, AutoMinorLocator
from mpl_toolkits.mplot3d import Axes3D


def one_d_hist_plot(fig, ax, bin_edges, counts, fit=None, count_zorder=[],
                    fit_zorder=[], count_labels=[], fit_labels=[], xlog=True,
                    ylog=True, xlabel='Amplitude', ylabel='Counts',
                    title='Visibility Difference Histogram'):

    bin_widths = np.diff(bin_edges)
    bin_centers = bin_edges[:-1] + 0.5 * bin_widths

    for i in range(len(count_labels)):
        ax.step(bin_edges, counts[i], where='pre', label=count_labels[i],
                zorder=count_zorder[i])

    if fit:
        bin_widths = np.diff(bin_edges)
        bin_centers = bin_edges[:-1] + 0.5 * bin_widths
        for i in range(len(fit_labels)):
            ax.plot(bin_centers, fit[i], label=fit_labels[i],
                    zorder=fit_zorder[i])

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


def line_plot(self, fig, ax, data, title='Visibility Difference Average',
              xlabel='Frequency (Mhz)', ylabel='Visibility Amplitude',
              zorder=[], data_labels=[], xticklabels=[]):

    for label in data:
        ax.plot(range(len(data[label])), data[label], label=label, zorder=zorder[label])

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.legend()
    if xticklabels:
        ax.set_xticklabels(xticklabels)


def image_plot(self, fig, ax, data, cmap, vmin=0, vmax=None, title='',
               aspect_ratio=3, xlabel='Frequency (Mhz)', ylabel='Time Pair',
               cbar_label='Counts RFI', xticks=[], yticks=[], xminors=None,
               yminors=None, xticklabels=None, yticklabels=None):

    data = np.ma.masked_equal(data, 0)
    cmap = cmap
    cmap.set_bad(color='white')

    if not vmax:
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


def scatter_plot(self, fig, ax, x_data, y_data, title='', xlabel='',
                 ylabel='', c=[]):

    if c:
        ax.scatter(x, y, c=c)
    else:
        ax.scatter(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)