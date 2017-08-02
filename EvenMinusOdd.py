import numpy as np
import matplotlib.pyplot as plt
import pyuvdata as pyuv
from matplotlib import cm
from math import floor, ceil, log10
from matplotlib.gridspec import GridSpec
import time
import os


class EvenMinusOdd:

    UV = pyuv.UVData()
    even = pyuv.UVData()
    odd = pyuv.UVData()
    EMO = pyuv.UVData()

    def __init__(self, BEF, TEF):

        self.BEF = BEF
        self.TEF = TEF

    def read_even_odd(self, filepath):

        self.UV.read_uvfits(filepath)

        self.even = self.UV.select(times=[self.UV.time_array[(2 * k + 1) * self.UV.Nbls]
                                   for k in range(self.UV.Ntimes / 2 - self.TEF)], inplace=False)
        self.odd = self.UV.select(times=[self.UV.time_array[2 * k * self.UV.Nbls]
                                  for k in range(self.TEF, self.UV.Ntimes / 2)], inplace=False)

        if self.BEF:
            coarse_width = 1.28 * 10**(6)  # coarse band width of MWA in hz
            Ncoarse = (self.UV.freq_array[0, -1] - self.UV.freq_array[0, 0]) / coarse_width
            Mcoarse = coarse_width / self.UV.channel_width  # Number of fine channels per coarse channel
            LEdges = [Mcoarse * p for p in range(Ncoarse)]
            REdges = [Mcoarse - 1 + Mcoarse * p for p in range(Ncoarse)]

            self.even.select(freq_chans=[x for x in range(self.UV.Nfreqs) if x not in
                             LEdges and x not in REdges])
            self.odd.select(freq_chans=[x for x in range(self.UV.Nfreqs) if x not in
                            LEdges and x not in REdges])

        self.EMO.data_array = self.even.data_array - self.odd.data_array

    def flag_operations(self, flag_slice='Unflagged'):

        if flag_slice is 'Unflagged':
            A = np.logical_and(np.logical_not(self.even.flag_array),
                               np.logical_not(self.odd.flag_array))
        elif flag_slice is 'And':
            A = np.logical_and(self.even.flag_array, self.odd.flag_array)
        elif flag_slice is 'XOR':
            A = np.logical_xor(self.even.flag_array, self.odd.flag_array)
        elif flag_slice is 'All':
            A = np.ones(self.even.flag_array.shape, dtype=bool)

        return(A)

    def one_d_hist_prepare(self, flag_slice='Unflagged'):
        N = np.prod(self.EMO.data_array.shape)
        data = np.reshape(self.EMO.data_array, N)
        flags = np.reshape(self.flag_operations(flag_slice=flag_slice), N)

        data = np.absolute(data)

        data = data[flags > 0]

        return(data)

    def one_d_hist_plot(self, fig, ax, data, bins, label, title, ylog=True, xlog=True):  # Data/title are tuples if multiple hists

        ax.hist(data, bins=bins, histtype='step', label=label)
        ax.set_title(title)

        if ylog:
            ax.set_yscale('log', nonposy='clip')
        else:
            ax.set_yscale('linear')

        if xlog:
            ax.set_xscale('log', nonposy='clip')
        else:
            ax.set_xscale('linear')

        ax.set_xlabel('Amplitude (' + self.even.vis_units + ')')
        ax.set_ylabel('Counts')
        ax.legend()

    def waterfall_hist_prepare(self, band, fraction=True, flag_slice='Unflagged'):  # band is a tuple (min,max)

        data = np.absolute(self.EMO.data_array)
        H = np.zeros([self.even.Ntimes, self.even.Nfreqs, self.even.Npols])

        flags = self.flag_operations(flag_slice=flag_slice)

        ind = np.where((min(band) < data) & (data < max(band)) & (flags > 0))  # Returns list of four-index combos
        IND0 = np.copy(ind[0])  # Following steps are to collapse blt into t (tuples cannot be assigned element-wise)
        IND0 = IND0 / self.UV.Nbls
        IND = (IND0, ind[2], ind[3])  # ind[1] contains only 0's (spectral window)
        for p in range(len(IND[0])):
            H[IND[0][p], IND[1][p], IND[2][p]] += 1

        if fraction is True:
            N = float(self.UV.Nbls * self.UV.Npols)
            H = H / N

        return(H)

    def waterfall_hist_plot(self, fig, ax, H, title, vmax, aspect_ratio=6, fraction=True):

        H = np.ma.masked_equal(H, 0)
        cmap = cm.plasma
        cmap.set_bad(color='white')

        cax = ax.imshow(H, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title)

        y_ticks = [self.even.Ntimes * k / 4 for k in range(5)]
        x_ticks = [self.even.Nfreqs * k / 6 for k in range(7)]
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_aspect(aspect_ratio)
        cbar = fig.colorbar(cax, ax=ax)
        if fraction:
            cbar.set_label('Fraction RFI')
        else:
            cbar.set_label('Counts RFI')

    def rfi_catalog(self, obslist, inpath, outpath, thresh_min=2000):  # obslist should be a list of integers (OBSID's)

        Nobs = len(obslist)

        for l in range(Nobs):
            if l % 1 == 0:
                print('Iteration ' + str(l) + ' started at ' + time.strftime('%H:%M:%S'))

            self.read_even_odd(inpath + str(obslist[l]) + '.uvfits')

            if l % 10 == 0:
                print('Finished reading at ' + time.strftime('%H:%M:%S'))

            AMPall = self.one_d_hist_prepare(flag_slice='All')
            AMPunflagged = self.one_d_hist_prepare(flag_slice='Unflagged')
            MAXAMP = max(AMPall)
            MINAMP = min(AMPall[np.nonzero(AMPall)])
            N_all = len(AMPall)
            N_unflagged = len(AMPunflagged)
            N0_all = N_all - np.count_nonzero(AMPall)
            N0_unflagged = N_unflagged - np.count_nonzero(AMPunflagged)

            AMPdata = [AMPunflagged, AMPall]
            AMPlabel = ['Unflagged', 'All']

            if l % 10 == 0:
                print('Finished preparing amplitude hist at ' + time.strftime('%H:%M:%S'))

            Wall = self.waterfall_hist_prepare((thresh_min, MAXAMP), flag_slice='All')
            Wunflagged = self.waterfall_hist_prepare((thresh_min, MAXAMP), flag_slice='Unflagged')

            W = [Wunflagged, Wall]

            MAXW_all_list = [np.amax(Wall[:, :, k]) for k in range(Wall.shape[2])]
            MAXW_all_auto = max(MAXW_all_list[0:2])
            MAXW_all_cross = max(MAXW_all_list[2:4])
            MAXW_all_list = [MAXW_all_auto, MAXW_all_auto, MAXW_all_cross, MAXW_all_cross]

            MAXW_unflagged_list = [np.amax(Wunflagged[:, :, k]) for k in range(Wunflagged.shape[2])]
            MAXW_unflagged_auto = max(MAXW_unflagged_list[0:2])
            MAXW_unflagged_cross = max(MAXW_unflagged_list[2:4])
            MAXW_unflagged_list = [MAXW_unflagged_auto, MAXW_unflagged_auto,
                                   MAXW_unflagged_cross, MAXW_unflagged_cross]

            MAXW_list = [MAXW_unflagged_list, MAXW_all_list]

            if l % 10 == 0:
                print('Finished preparing the waterfall hist at ' + time.strftime('%H:%M:%S'))

            figs = [plt.figure(figsize=(14, 8)), plt.figure(figsize=(14, 8))]
            gs = GridSpec(3, 2)
            axes = [[figs[0].add_subplot(gs[1, 0]), figs[0].add_subplot(gs[1, 1]), figs[0].add_subplot(gs[2, 0]),
                    figs[0].add_subplot(gs[2, 1]), figs[0].add_subplot(gs[0, :])],
                    [figs[1].add_subplot(gs[1, 0]), figs[1].add_subplot(gs[1, 1]), figs[1].add_subplot(gs[2, 0]),
                     figs[1].add_subplot(gs[2, 1]), figs[1].add_subplot(gs[0, :])]]

            for x in figs:
                x.subplots_adjust(left=0.13, bottom=0.11, right=0.90, top=0.88,
                                  wspace=0.20, hspace=0.46)

            keys = [-8 + k for k in range(13)]
            keys.remove(0)
            values = ['YX', 'XY', 'YY', 'XX', 'LR', 'RL', 'LL', 'RR', 'I', 'Q', 'U', 'V']

            pol_titles = dict(zip(keys, values))
            flag_titles = ['Unflagged', 'All']

            def sigfig(x, s=4):  # s is number of sig-figs
                if x == 0:
                    return(0)
                else:
                    n = int(floor(log10(abs(x))))
                    y = 10**n * round(10**(-n) * x, s - 1)
                    return(y)

            for m in range(2):
                for n in range(5):
                    if n < 4:
                        self.waterfall_hist_plot(figs[m], axes[m][n], W[m][:, :, n],
                                                 pol_titles[self.UV.polarization_array[n]] +
                                                 ' ' + flag_titles[m], MAXW_list[m][n])
                        if n in [0, 2]:  # Some get axis labels others do not
                            axes[m][n].set_ylabel('Time Pair')
                        if n in [2, 3]:
                            axes[m][n].set_xlabel('Frequency (Mhz)')
                            x_ticks_labels = [str(sigfig(self.UV.freq_array[0,
                                              self.even.Nfreqs * k / 6] *
                                              10**(-6))) for k in range(6)]
                            x_ticks_labels.append(str(sigfig((self.UV.freq_array[0, -1] +
                                                  self.UV.channel_width) * 10**(-6))))
                            axes[m][n].set_xticklabels(x_ticks_labels)
                        if n in [0, 1]:
                            axes[m][n].set_xticklabels([])
                        if n in [1, 3]:
                            axes[m][n].set_yticklabels([])
                    else:
                        bins = np.logspace(log10(MINAMP), log10(MAXAMP), num=1001)
                        self.one_d_hist_plot(figs[m], axes[m][n], AMPdata,
                                             bins, AMPlabel, 'RFI Catalog ' +
                                             str(obslist[l]))
                        axes[m][n].axvline(x=thresh_min, color='r')

                figs[m].savefig(outpath + str(obslist[l]) + '_RFI_Diagnostic_' +
                                flag_titles[m] + '.png')

            if l % 10 == 0:
                print('Figure saved! ' + time.strftime('%H:%M:%S'))
