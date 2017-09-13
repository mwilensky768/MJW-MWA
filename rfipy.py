import numpy as np
import pyuvdata as pyuv
from matplotlib import cm, use
use('Agg')
import matplotlib.pyplot as plt
from math import floor, ceil, log10
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FixedLocator, AutoMinorLocator
import time
import os
from scipy.stats import rayleigh


class RFI:

    UV = pyuv.UVData()
    data = []

    def read_even_odd(self, filepath, bad_time_indices=[], coarse_band_remove=False):  # specify time indices to remove and filepath

        self.UV.read_uvfits(filepath)
        times = [self.UV.time_array[k * self.UV.Nbls] for k in range(self.UV.Ntimes)]
        bad_times = []
        for k in bad_time_indices:
            bad_times.append(times[k])
        for bad_time in bad_times:
            times.remove(bad_time)
        self.UV.select(times=times)

        if coarse_band_remove:
            coarse_width = 1.28 * 10**(6)  # coarse band width of MWA in hz
            Ncoarse = (self.UV.freq_array[0, -1] - self.UV.freq_array[0, 0]) / coarse_width
            Mcoarse = coarse_width / self.UV.channel_width  # Number of fine channels per coarse channel
            LEdges = [Mcoarse * p for p in range(Ncoarse)]
            REdges = [Mcoarse - 1 + Mcoarse * p for p in range(Ncoarse)]

            self.UV.select(freq_chans=[x for x in range(self.UV.Nfreqs) if x not in
                                       LEdges and x not in REdges])

    def data_prepare(self):
        self.data = np.absolute(np.diff(np.reshape(self.UV.data_array,
                                [self.UV.Ntimes, self.UV.Nbls, self.UV.Nspws,
                                 self.UV.Nfreqs, self.UV.Npols]), axis=0))

    def flag_operations(self, flag_slice='Unflagged'):

        A = np.reshape(self.UV.flag_array, [self.UV.Ntimes, self.UV.Nbls, self.UV.Nspws,
                       self.UV.Nfreqs, self.UV.Npols])

        if flag_slice is 'Unflagged':
            A = np.logical_and(np.logical_not(A[0:(self.UV.Ntimes - 1), :, :, :, :]),
                               np.logical_not(A[1:self.UV.Ntimes, :, :, :, :]))
        elif flag_slice is 'And':
            A = np.logical_and(A[0:(self.UV.Ntimes - 1), :, :, :, :], A[1:self.UV.Ntimes, :, :, :, :])
        elif flag_slice is 'XOR':
            A = np.logical_xor(A[0:(self.UV.Ntimes - 1), :, :, :, :], A[1:self.UV.Ntimes, :, :, :, :])
        elif flag_slice is 'All':
            A = np.ones([self.UV.Ntimes - 1, self.UV.Nbls, self.UV.Nspws,
                         self.UV.Nfreqs, self.UV.Npols], dtype=bool)

        return(A)

    def one_d_hist_prepare(self, flag_slice='Unflagged', time_drill=[], freq_drill=[],
                           time_slice=[], freq_slice=[]):

        flags = self.flag_operations(flag_slice=flag_slice)
        values = self.data

        if time_drill:
            values = self.data[time_drill, :, :, :, :]
            flags = flags[time_drill, :, :, :, :]
        if time_slice:
            values = self.data[min(time_slice):max(time_slice) + 1, :, :, :, :]
            flags = flags[min(time_slice):max(time_slice) + 1, :, :, :, :]
        if freq_drill:
            values = self.data[:, :, :, freq_drill, :]
            flags = flags[:, :, :, freq_drill, :]
        if freq_slice:
            values = self.data[:, :, :, min(freq_slice):max(freq_slice) + 1, :]
            flags - flags[:, :, :, min(freq_slice):max(freq_slice) + 1, :]

        N = np.prod(values.shape)
        values = np.reshape(values, N)
        flags = np.reshape(flags, N)

        values = values[flags > 0]

        return(values)

    def one_d_hist_plot(self, fig, ax, data, label, title, bins=np.logspace(-3, 5, num=1001), writepath='',
                        ylog=True, xlog=True, write=False):  # Data/title are tuples if multiple hists

        n, bins, patches = ax.hist(data, bins=bins, histtype='step', label=label)
        if write:
            np.save(writepath, n[0])
        ax.set_title(title)

        if ylog:
            ax.set_yscale('log', nonposy='clip')
        else:
            ax.set_yscale('linear')

        if xlog:
            ax.set_xscale('log', nonposy='clip')
        else:
            ax.set_xscale('linear')

        ax.set_xlabel('Amplitude (' + self.UV.vis_units + ')')
        ax.set_ylabel('Counts')
        ax.legend()

    def waterfall_hist_prepare(self, band, plot_type='time-freq', fraction=True,
                               flag_slice='Unflagged'):  # band is a tuple (min,max)

        flags = np.reshape(self.flag_operations(flag_slice=flag_slice), self.data.shape)

        ind = np.where((min(band) < self.data) & (self.data < max(band)) & (flags > 0))  # Returns list of five-index combos

        if plot_type == 'time-freq':
            H = np.zeros([self.UV.Ntimes - 1, self.UV.Nfreqs, self.UV.Npols])
            for p in range(len(ind[0])):
                H[ind[0][p], ind[3][p], ind[4][p]] += 1
            N = float(self.UV.Nbls * self.UV.Npols)
            if fraction:
                N = float(self.UV.Nbls * self.UV.Npols)
                H = H / N
            return(H)
        elif plot_type == 'ant-freq':
            unique_times = np.unique(ind[0])
            N_unique_times = len(unique_times)
            H = np.zeros([self.UV.Nants_telescope, self.UV.Nfreqs, self.UV.Npols,
                          N_unique_times])
            ant1_ind = []
            ant2_ind = []
            for inds in ind[1]:
                ant1_ind.append(self.UV.ant_1_array[inds])
                ant2_ind.append(self.UV.ant_2_array[inds])
            ant_ind = [np.array(ant1_ind), np.array(ant2_ind)]
            for p in range(2):
                for q in range(len(ind[0])):
                    H[ant_ind[p][q], ind[3][q], ind[4][q],
                      np.where(unique_times == ind[0][q])[0][0]] += 1
            return(H, unique_times)
        elif plot_type == 'ant-time':
            unique_freqs = np.unique(ind[3])
            N_unique_freqs = len(unique_freqs)
            H = np.zeros([self.UV.Nants_telescope, self.UV.Ntimes - 1, self.UV.Npols,
                          N_unique_freqs])
            ant1_ind = []
            ant2_ind = []
            for inds in ind[1]:
                ant1_ind.append(self.UV.ant_1_array[inds])
                ant2_ind.append(self.UV.ant_2_array[inds])
            ant_ind = [np.array(ant1_ind), np.array(ant2_ind)]
            for p in range(2):
                for q in range(len(ind[3])):
                    H[ant_ind[p, q], ind[0][q], ind[4][q],
                      np.where(unique_freqs == ind[3][q])[0][0]] += 1
            return(H, unique_freqs)

    def waterfall_hist_plot(self, fig, ax, H, title, vmax, aspect_ratio=3, fraction=True):

        H = np.ma.masked_equal(H, 0)
        cmap = cm.cool
        cmap.set_bad(color='white')

        cax = ax.imshow(H, cmap=cmap, vmin=0, vmax=vmax)
        ax.set_title(title)

        y_ticks = [(H.shape[0]) * k / 5 for k in range(5)]
        y_ticks.append(H.shape[0] - 1)
        y_minors = range(H.shape[0])
        for y in y_ticks:
            y_minors.remove(y)
        y_minor_locator = FixedLocator(y_minors)
        x_ticks = [H.shape[1] * k / 6 for k in range(6)]
        x_ticks.append(H.shape[1] - 1)
        x_minor_locator = AutoMinorLocator(4)
        ax.set_xticks(x_ticks)
        ax.xaxis.set_minor_locator(x_minor_locator)
        ax.set_yticks(y_ticks)
        ax.set_aspect(aspect_ratio)
        ax.yaxis.set_minor_locator(y_minor_locator)
        cbar = fig.colorbar(cax, ax=ax)
        if fraction:
            cbar.set_label('Fraction RFI')
        else:
            cbar.set_label('Counts RFI')

    def rfi_catalog(self, obs, inpath, outpath, bad_time_indices=[],
                    coarse_band_remove=False, band=(2000, 10**5), hist_write=False,
                    hist_write_path=''):

        print('Started at ' + time.strftime('%H:%M:%S'))

        self.read_even_odd(inpath, bad_time_indices=bad_time_indices,
                           coarse_band_remove=coarse_band_remove)
        self.data_prepare()

        print('Finished reading at ' + time.strftime('%H:%M:%S'))

        AMP = [self.one_d_hist_prepare(flag_slice='Unflagged'),
               self.one_d_hist_prepare(flag_slice='All')]

        print('Finished preparing amplitude hist at ' + time.strftime('%H:%M:%S'))

        gs = GridSpec(3, 2)
        gs_loc = [[1, 0], [1, 1], [2, 0], [2, 1]]

        pol_keys = [-8 + k for k in range(13)]
        pol_keys.remove(0)
        pol_values = ['YX', 'XY', 'YY', 'XX', 'LR', 'RL', 'LL', 'RR', 'I', 'Q', 'U', 'V']

        pol_titles = dict(zip(pol_keys, pol_values))
        flag_slices = ['Unflagged', 'All']

        def sigfig(x, s=4):  # s is number of sig-figs
            if x == 0:
                return(0)
            else:
                n = int(floor(log10(np.absolute(x))))
                y = 10**n * round(10**(-n) * x, s - 1)
                return(y)

        for flag_slice in flag_slices:
            W = self.waterfall_hist_prepare(band, plot_type='time-freq',
                                            fraction=True, flag_slice=flag_slice)

            MAXW_list = [np.amax(W[:, :, k]) for k in range(W.shape[2])]
            MAXW_auto = max(MAXW_list[0:2])
            MAXW_cross = max(MAXW_list[2:4])
            MAXW_list = [MAXW_auto, MAXW_auto, MAXW_cross, MAXW_cross]

            fig = plt.figure(figsize=(14, 8))
            ax = fig.add_subplot(gs[0, :])
            self.one_d_hist_plot(fig, ax, AMP, ['Unflagged', 'All'], ' RFI Catalog ' + str(obs))
            ax.axvline(x=min(band), color='r')
            for n in range(self.UV.Npols):
                ax = fig.add_subplot(gs[gs_loc[n][0], gs_loc[n][1]])
                self.waterfall_hist_plot(fig, ax, W[:, :, n],
                                         pol_titles[self.UV.polarization_array[n]] +
                                         ' ' + flag_slice, MAXW_list[n])

                ax.set_ylabel('Time Pair')
                ax.set_xlabel('Frequency (Mhz)')
                x_ticks_labels = [str(sigfig(self.UV.freq_array[0, self.UV.Nfreqs * k / 6] *
                                  10**(-6))) for k in range(6)]
                x_ticks_labels.append(str(sigfig((self.UV.freq_array[0, -1] * 10**(-6)))))
                ax.set_xticklabels(x_ticks_labels)

            plt.tight_layout()
            fig.savefig(outpath + str(obs) + '_RFI_Diagnostic_' + flag_slice + '.png')
            plt.close(fig)

    def catalog_drill(self, obs, inpath, outpath, plot_type, bad_time_indices=[],
                      coarse_band_remove=False, band=(2000, 100000)):

        self.read_even_odd(inpath, bad_time_indices=bad_time_indices,
                           coarse_band_remove=coarse_band_remove)
        self.data_prepare()

        flag_slices = ['All', 'Unflagged']

        pol_keys = [-8 + k for k in range(13)]
        pol_keys.remove(0)
        pol_values = ['YX', 'XY', 'YY', 'XX', 'LR', 'RL', 'LL', 'RR', 'I', 'Q', 'U', 'V']
        pol_titles = dict(zip(pol_keys, pol_values))

        plot_type_keys = ['ant-freq', 'ant-time']
        plot_type_title_values = [' t = ', ' f = ']
        x_label_values = ['Frequency (Mhz)', 'Time-Pair']

        plot_type_titles = dict(zip(plot_type_keys, plot_type_title_values))
        x_labels = dict(zip(plot_type_keys, x_label_values))

        def sigfig(x, s=4):  # s is number of sig-figs
            if x == 0:
                return(0)
            else:
                n = int(floor(log10(np.absolute(x))))
                y = 10**n * round(10**(-n) * x, s - 1)
                return(y)

        gs = GridSpec(3, 2)
        gs_loc = [[1, 0], [1, 1], [2, 0], [2, 1]]

        for flag_slice in flag_slices:
            W, uniques = self.waterfall_hist_prepare(band, plot_type=plot_type,
                                                     fraction=False, flag_slice=flag_slice)
            if plot_type == 'ant-time':
                unique_freqs = [self.UV.freq_array[0, m] for m in uniques]
            N_events = W.shape[3]
            for k in range(N_events):
                fig = plt.figure(figsize=(14, 8))
                ax = fig.add_subplot(gs[0, :])
                if plot_type == 'ant-freq':
                    AMP = [self.one_d_hist_prepare(flag_slice='Unflagged', time_drill=uniques[k]),
                           self.one_d_hist_prepare(flag_slice='All', time_drill=uniques[k])]
                else:
                    AMP = [self.one_d_hist_prepare(flag_slice='Unflagged', freq_drill=uniques[k]),
                           self.one_d_hist_prepare(flag_slice='All', freq_drill=uniques[k])]
                self.one_d_hist_plot(fig, ax, AMP, ['Unflagged', 'All'], str(obs) + ' Drill ' +
                                     plot_type_titles[plot_type] + str(unique_freqs[k]))
                ax.axvline(x=min(band), color='r')
                for l in range(self.UV.Npols):
                    vmax = np.amax(W[:, :, l, k])
                    ax = fig.add_subplot(gs[gs_loc[l][0], gs_loc[l][1]])
                    self.waterfall_hist_plot(fig, ax, W[:, :, l, k],
                                             'Drill ' + pol_titles[self.UV.polarization_array[l]] +
                                             ' ' + flag_slice,
                                             vmax, aspect_ratio=1, fraction=False)
                    ax.set_ylabel('Antenna #')
                    ax.set_xlabel(x_labels[plot_type])
                    if plot_type == 'ant-freq':
                        x_tick_labels = [str(sigfig(self.UV.freq_array[0, self.UV.Nfreqs * m / 6] *
                                             10**(-6))) for m in range(6)]  # for 'ant-freq' only
                        x_tick_labels.append(str(sigfig((self.UV.freq_array[0, -1] * 10**(-6)))))
                        ax.set_xticklabels(x_tick_labels)
                plt.tight_layout()
                fig.savefig(outpath + str(obs) + '_Drill_' + flag_slice + '_' +
                            str(uniques[k]) + '.png')
                plt.close(fig)
