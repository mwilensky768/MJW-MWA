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
from scipy.optimize import curve_fit


class RFI:

    def __init__(self, obs, filepath, bad_time_indices=[0, -3, -2, -1], coarse_band_remove=False,
                 auto_remove=True, filetype='uvfits'):
        self.obs = obs
        self.UV = pyuv.UVData()
        if filetype is 'uvfits':
            self.UV.read_uvfits(filepath)
        elif filetype is 'miriad':
            self.UV.read_miriad(filepath)

        if bad_time_indices:
            times = [self.UV.time_array[k * self.UV.Nbls] for k in range(self.UV.Ntimes)]
            bad_times = []
            for k in bad_time_indices:
                bad_times.append(times[k])
            for bad_time in bad_times:
                times.remove(bad_time)
            self.UV.select(times=times)

        if auto_remove:
            blt_inds = [k for k in range(self.UV.Nblts) if
                        self.UV.ant_1_array[k] != self.UV.ant_2_array[k]]
            self.UV.select(blt_inds=blt_inds)

        if coarse_band_remove:  # MWA specific
            coarse_width = 1.28 * 10**(6)  # coarse band width of MWA in hz
            Ncoarse = (self.UV.freq_array[0, -1] - self.UV.freq_array[0, 0]) / coarse_width
            Mcoarse = coarse_width / self.UV.channel_width  # Number of fine channels per coarse channel
            LEdges = [Mcoarse * p for p in range(Ncoarse)]
            REdges = [Mcoarse - 1 + Mcoarse * p for p in range(Ncoarse)]

            self.UV.select(freq_chans=[x for x in range(self.UV.Nfreqs) if x not in
                                       LEdges and x not in REdges])

        self.data_array = np.diff(np.reshape(self.UV.data_array,
                                  [self.UV.Ntimes, self.UV.Nbls, self.UV.Nspws,
                                   self.UV.Nfreqs, self.UV.Npols]), axis=0)

    def flag_operations(self, flag_slice='Unflagged', coarse_band_ignore=False):

        A = np.reshape(self.UV.flag_array, [self.UV.Ntimes, self.UV.Nbls, self.UV.Nspws,
                       self.UV.Nfreqs, self.UV.Npols])

        if flag_slice is 'Unflagged':
            A = np.logical_not(np.logical_or(A[0:(self.UV.Ntimes - 1), :, :, :, :],
                                             A[1:self.UV.Ntimes, :, :, :, :]))
        elif flag_slice is 'Flagged':
            A = np.logical_or(A[0:(self.UV.Ntimes - 1), :, :, :, :], A[1:self.UV.Ntimes, :, :, :, :])
        elif flag_slice is 'And':
            A = np.logical_and(A[0:(self.UV.Ntimes - 1), :, :, :, :], A[1:self.UV.Ntimes, :, :, :, :])
        elif flag_slice is 'XOR':
            A = np.logical_xor(A[0:(self.UV.Ntimes - 1), :, :, :, :], A[1:self.UV.Ntimes, :, :, :, :])
        elif flag_slice is 'All':
            A = np.ones([self.UV.Ntimes - 1, self.UV.Nbls, self.UV.Nspws,
                         self.UV.Nfreqs, self.UV.Npols], dtype=bool)

        if coarse_band_ignore:
            coarse_width = 1.28 * 10**(6)  # coarse band width of MWA in hz
            Ncoarse = (self.UV.freq_array[0, -1] - self.UV.freq_array[0, 0]) / coarse_width
            Mcoarse = coarse_width / self.UV.channel_width  # Number of fine channels per coarse channel
            LEdges = [Mcoarse * p for p in range(Ncoarse)]
            REdges = [Mcoarse - 1 + Mcoarse * p for p in range(Ncoarse)]

            for x in LEdges + REdges:
                A[:, :, :, x, :] = 0

        return(A)

    def one_d_hist_prepare(self, flag_slice='Unflagged', time_drill=[], freq_drill=[],
                           time_slice=[], freq_slice=[], coarse_band_ignore=False,
                           bins='auto', fit=False, fit_window=[0, 10**12], write=False,
                           writepath='', temp_write=False):

        flags = self.flag_operations(flag_slice=flag_slice,
                                     coarse_band_ignore=coarse_band_ignore)
        values = np.absolute(self.data_array)

        if time_drill:
            values = values[time_drill, :, :, :, :]
            flags = flags[time_drill, :, :, :, :]
        if time_slice:
            values = values[min(time_slice):max(time_slice), :, :, :, :]
            flags = flags[min(time_slice):max(time_slice), :, :, :, :]
        if freq_drill:
            values = values[:, :, :, freq_drill, :]
            flags = flags[:, :, :, freq_drill, :]
        if freq_slice:
            values = values[:, :, :, min(freq_slice):max(freq_slice), :]
            flags = flags[:, :, :, min(freq_slice):max(freq_slice), :]

        if bins is 'auto':
            MIN = np.amin(values[values > 0])
            MAX = np.amax(values)
            bins = np.logspace(floor(log10(MIN)), ceil(log10(MAX)), num=1001)
        else:
            bins = bins

        bin_widths = np.diff(bins)
        bin_centers = bins[:-1] + 0.5 * bin_widths

        if fit:
            fit = np.zeros(len(bins) - 1)
            m = np.copy(fit)
            if temp_write:
                pol_keys = [-8 + k for k in range(13)]
                pol_keys.remove(0)
                pol_values = ['YX', 'XY', 'YY', 'XX', 'LR', 'RL', 'LL', 'RR',
                              'I', 'Q', 'U', 'V']
                pol_titles = dict(zip(pol_keys, pol_values))
                sigma_array = np.zeros(values.shape[3])
            for l in range(values.shape[4]):
                for k in range(values.shape[3]):
                    N = np.prod(values.shape[:3])
                    temp_values = values[:, :, :, k, l]
                    temp_flags = flags[:, :, :, k, l]
                    temp_values = np.reshape(temp_values, N)
                    temp_flags = np.reshape(temp_flags, N)
                    temp_values = temp_values[temp_flags > 0]
                    n, bins = np.histogram(temp_values, bins=bins)
                    m += n
                    bin_cond = np.logical_and(min(fit_window) < n, n < max(fit_window))
                    bin_window = bins[:-1][bin_cond]
                    if len(bin_window > 0):
                        data_cond = np.logical_and(min(bin_window) < temp_values,
                                                   temp_values < max(bin_window))
                        N_fit = len(temp_values[data_cond])
                        if N_fit > 0:
                            sigma = np.sqrt(0.5 * np.sum(temp_values[data_cond]**2) / N_fit)
                            fit += N * bin_widths * (1 / sigma**2) * bin_centers * \
                                np.exp(-bin_centers**2 / (2 * sigma ** 2))
                        elif temp_write:
                            sigma = 0
                            sigma_array[k] = sigma
                    elif temp_write:
                        sigma = 0
                        sigma_array[k] = sigma
                if temp_write:
                    np.save(writepath + self.obs + '_sigma_' +
                            pol_titles[self.UV.polarization_array[l]] + '.npy',
                            sigma_array)

        else:
            N = np.prod(values.shape)
            values = np.reshape(values, N)
            flags = np.reshape(flags, N)
            values = values[flags > 0]
            m, bins = np.histogram(values, bins=bins)
            fit = [0, ]

        if write:
            np.save(writepath + self.obs + '_hist.npy', m)
            np.save(writepath + self.obs + '_bins.npy', bins)
            np.save(writepath + self.obs + '_fit.npy', fit)

        return({flag_slice: (m, bins, fit)})

    def one_d_hist_plot(self, fig, ax, data, title, ylog=True, xlog=True, res_ax=[]):  # Data/title are tuples if multiple hists

        zorder = {'Unflagged': 8, 'Flagged': 6, 'And': 4, 'XOR': 2, 'All': 0}

        for x in data:
            break

        bin_widths = np.diff(data[x][1])
        bin_centers = data[x][1][:-1] + 0.5 * bin_widths
        for label in data:
            ax.step(data[label][1][:-1], data[label][0], where='pre', label=label,
                    zorder=zorder[label])
            if len(data[label][2]) > 1:
                ax.plot(bin_centers, data[label][2], label='Fit', zorder=10)
                if res_ax:
                    residual = data[label][0] - data[label][2]
                    if np.all(data[label][2][data[label][1] > 0] > 0):
                        chi_square = np.sum((residual**2) / data[label][2]) / (len(data[label][1]) - 1)
                        res_label = 'Residual: chi_square/DoF = ' + str(chi_square)
                    else:
                        res_label = 'Residual'
                    res_ax.plot(bin_centers, residual, label='Residual')
                    res_ax.set_xscale('log', nonposy='clip')
                    res_ax.set_yscale('linear')
                    res_ax.legend()

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
        ax.set_ylim([10**(-1), 10 * max([np.amax(data[x][0]) for x in data])])
        ax.legend()

    def waterfall_hist_prepare(self, band, plot_type='time-freq', fraction=True,
                               flag_slice='Unflagged', coarse_band_ignore=False):  # band is a tuple (min,max)

        flags = np.reshape(self.flag_operations(flag_slice=flag_slice,
                                                coarse_band_ignore=coarse_band_ignore),
                           self.data_array.shape)

        values = np.absolute(self.data_array)

        ind = np.where((min(band) < values) & (values < max(band)) & (flags > 0))  # Returns list of five-index combos

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
                    H[ant_ind[p][q], ind[0][q], ind[4][q],
                      np.where(unique_freqs == ind[3][q])[0][0]] += 1
            return(H, unique_freqs)

    def ant_pol_prepare(self, time, freq):

        dim = np.sqrt(self.UV.Npols) * self.UV.Nants_telescope

        T = np.zeros([dim, dim])

        q = [[0, 0], [self.UV.Nants_telescope, self.UV.Nants_telescope],
             [0, self.UV.Nants_telescope], [self.UV.Nants_telescope, 0]]

        for m in range(self.UV.Nbls):
            for n in range(self.UV.Npols):
                T[self.UV.ant_1_array[m] + q[n][0], self.UV.ant_2_array[m] +
                  q[n][1]] = np.imag(self.data_array[time, m, 0, freq, n])
                T[self.UV.ant_2_array[m] + q[n][0], self.UV.ant_1_array[m] +
                  q[n][1]] = np.real(self.data_array[time, m, 0, freq, n])

        return(T)

    def image_plot(self, fig, ax, H, title, vmin, vmax, aspect_ratio=3,
                   fraction=True, y_type='time', x_type='freq'):

        def sigfig(x, s=4):  # s is number of sig-figs
            if x == 0:
                return(0)
            else:
                n = int(floor(log10(np.absolute(x))))
                y = 10**n * round(10**(-n) * x, s - 1)
                return(y)

        H = np.ma.masked_equal(H, 0)
        cmap = cm.cool
        cmap.set_bad(color='white')

        cax = ax.imshow(H, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)

        ticks = {'time': [(self.UV.Ntimes - 1) * k / 5 for k in range(5)],
                 'freq': [self.UV.Nfreqs * k / 6 for k in range(6)],
                 'ant': [self.UV.Nants_telescope * k / 4 for k in range(4)],
                 'ant-pol': [self.UV.Nants_telescope * k / 4 for k in range(8)]}
        ticks['time'].append(self.UV.Ntimes - 2)
        ticks['freq'].append(self.UV.Nfreqs - 1)
        ticks['ant'].append(self.UV.Nants_telescope - 1)
        ticks['ant-pol'].append(2 * self.UV.Nants_telescope - 1)

        minor_ticks = {'time': range(self.UV.Ntimes), 'freq': AutoMinorLocator(4),
                       'ant': AutoMinorLocator(8), 'ant-pol': AutoMinorLocator(8)}
        for tick in ticks['time']:
            minor_ticks['time'].remove(tick)
        minor_ticks['time'] = FixedLocator(minor_ticks['time'])

        ax.set_xticks(ticks[x_type])
        ax.xaxis.set_minor_locator(minor_ticks[x_type])
        ax.set_yticks(ticks[y_type])
        ax.yaxis.set_minor_locator(minor_ticks[y_type])

        if y_type == 'freq':
            y_tick_labels = [str(sigfig(self.UV.freq_array[0, k]) * 10 ** (-6)) for k in ticks[y_type]]
            ax.set_yticklabels(y_tick_labels)
        elif y_type == 'ant-pol':
            y_tick_labels = np.mod(ticks[y_type], self.UV.Nants_telescope)
            ax.set_yticklabels(y_tick_labels)
        if x_type == 'freq':
            x_tick_labels = [str(sigfig(self.UV.freq_array[0, k]) * 10 ** (-6)) for k in ticks[x_type]]
            ax.set_xticklabels(x_tick_labels)
        elif x_type == 'ant-pol':
            x_tick_labels = np.mod(ticks[x_type], self.UV.Nants_telescope)
            ax.set_xticklabels(x_tick_labels)

        x_labels = {'time': 'Time Pair', 'freq': 'Frequency (Mhz)', 'ant': 'Antenna Index',
                    'ant-pol': 'Antenna 2 Index'}
        y_labels = {'time': 'Time Pair', 'freq': 'Frequency (Mhz)', 'ant': 'Antenna Index',
                    'ant-pol': 'Antenna 1 Index'}

        ax.set_xlabel(x_labels[x_type])
        ax.set_ylabel(y_labels[y_type])

        ax.set_aspect(aspect_ratio)

        cbar = fig.colorbar(cax, ax=ax)
        if y_type == 'ant-pol':
            cbar.set_label(self.UV.vis_units)
        elif fraction:
            cbar.set_label('Fraction RFI')
        else:
            cbar.set_label('Counts RFI')

    def rfi_catalog(self, outpath, band=(2000, 10**5), hist_write=False,
                    hist_write_path='', fit=False, bins='auto',
                    flag_slices=['Unflagged', 'All'], coarse_band_ignore=False):

        def sigfig(x, s=4):  # s is number of sig-figs
            if x == 0:
                return(0)
            else:
                n = int(floor(log10(np.absolute(x))))
                y = 10**n * round(10**(-n) * x, s - 1)
                return(y)

        Amp = {}
        for flag_slice in flag_slices:
            Amp.update(self.one_d_hist_prepare(flag_slice=flag_slice, write=hist_write,
                                               coarse_band_ignore=coarse_band_ignore,
                                               writepath=hist_write_path, fit=fit,
                                               bins=bins))

        if self.UV.Npols > 1:
            gs = GridSpec(3, 2)
            gs_loc = [[1, 0], [1, 1], [2, 0], [2, 1]]
        else:
            gs = GridSpec(2, 1)
            gs_loc = [[1, 0], ]

        pol_keys = [-8 + k for k in range(13)]
        pol_keys.remove(0)
        pol_values = ['YX', 'XY', 'YY', 'XX', 'LR', 'RL', 'LL', 'RR', 'I', 'Q', 'U', 'V']

        pol_titles = dict(zip(pol_keys, pol_values))

        for flag_slice in flag_slices:
            W = self.waterfall_hist_prepare(band, plot_type='time-freq',
                                            fraction=True, flag_slice=flag_slice,
                                            coarse_band_ignore=coarse_band_ignore)

            if self.UV.Npols > 1:
                MAXW_list = range(4)
                MAXW_list[:2] = max([np.amax(W[:, :, k]) for k in [0, 1]])
                MAXW_list[2:4] = max([np.amax(W[:, :, k]) for k in [2, 3]])

                MINW_list = range(4)
                MINW_list[:2] = min([np.amin(W[:, :, k]) for k in [0, 1]])
                MINW_list[2:4] = min([np.amin(W[:, :, k]) for k in [2, 3]])
            else:
                MAXW_list = [np.amax(W[:, :, 0]), ]
                MINW_list = [np.amin(W[:, :, 0]), ]

            fig = plt.figure(figsize=(14, 8))
            ax = fig.add_subplot(gs[0, :])
            self.one_d_hist_plot(fig, ax, Amp, ' RFI Catalog ' + self.obs)
            ax.axvline(x=min(band), color='black')
            ax.axvline(x=max(band), color='black')
            for n in range(self.UV.Npols):
                ax = fig.add_subplot(gs[gs_loc[n][0], gs_loc[n][1]])
                self.image_plot(fig, ax, W[:, :, n],
                                pol_titles[self.UV.polarization_array[n]] +
                                ' ' + flag_slice, MINW_list[n], MAXW_list[n])

            plt.tight_layout()
            fig.savefig(outpath + self.obs + '_RFI_Diagnostic_' + flag_slice + '.png')
            plt.close(fig)

    def catalog_drill(self, outpath, plot_type='ant-freq', band=(2000, 10**5),
                      fit=False, bins='auto', flag_slices=['Unflagged', 'All'],
                      coarse_band_ignore=False):

        pol_keys = [-8 + k for k in range(13)]
        pol_keys.remove(0)
        pol_values = ['YX', 'XY', 'YY', 'XX', 'LR', 'RL', 'LL', 'RR', 'I', 'Q', 'U', 'V']
        pol_titles = dict(zip(pol_keys, pol_values))

        plot_type_keys = ['ant-freq', 'ant-time']
        aspect_values = [1, 0.2]
        x_type_values = ['freq', 'time']
        plot_type_title_values = [' t = ', ' f = ']
        x_label_values = ['Frequency (Mhz)', 'Time-Pair']
        path_label_values = ['t', 'f']

        aspect = dict(zip(plot_type_keys, aspect_values))
        x_type = dict(zip(plot_type_keys, x_type_values))
        plot_type_titles = dict(zip(plot_type_keys, plot_type_title_values))
        x_labels = dict(zip(plot_type_keys, x_label_values))
        path_labels = dict(zip(plot_type_keys, path_label_values))

        def sigfig(x, s=4):  # s is number of sig-figs
            if x == 0:
                return(0)
            else:
                n = int(floor(log10(np.absolute(x))))
                y = 10**n * round(10**(-n) * x, s - 1)
                return(y)

        if self.UV.Npols > 1:
            gs = GridSpec(3, 2)
            gs_loc = [[1, 0], [1, 1], [2, 0], [2, 1]]
        else:
            gs = GridSpec(2, 1)
            gs_loc = [[1, 0], ]

        for flag_slice in flag_slices:
            W, uniques = self.waterfall_hist_prepare(band, plot_type=plot_type,
                                                     fraction=False,
                                                     flag_slice=flag_slice,
                                                     coarse_band_ignore=coarse_band_ignore)
            N_events = W.shape[3]
            for k in range(N_events):

                fig = plt.figure(figsize=(14, 8))
                ax = fig.add_subplot(gs[0, :])

                if plot_type == 'ant-freq':
                    Amp = {}
                    for flag in flag_slices:
                        Amp.update(self.one_d_hist_prepare(flag_slice=flag,
                                                           time_drill=uniques[k],
                                                           coarse_band_ignore=coarse_band_ignore,
                                                           fit=fit, bins=bins))
                    self.one_d_hist_plot(fig, ax, Amp, self.obs + ' Drill ' +
                                         plot_type_titles[plot_type] +
                                         str(uniques[k]))
                elif plot_type == 'ant-time':
                    unique_freqs = [sigfig(self.UV.freq_array[0, m]) * 10**(-6) for
                                    m in uniques]
                    Amp = {}
                    for flag in flag_slices:
                        Amp.update(self.one_d_hist_prepare(flag_slice=flag,
                                                           freq_drill=uniques[k],
                                                           coarse_band_ignore=coarse_band_ignore,
                                                           fit=fit, bins=bins))
                    self.one_d_hist_plot(fig, ax, Amp, self.obs + ' Drill ' +
                                         plot_type_titles[plot_type] +
                                         str(unique_freqs[k]))
                ax.axvline(x=min(band), color='black')
                ax.axvline(x=max(band), color='black')

                for l in range(self.UV.Npols):
                    vmax = np.amax(W[:, :, l, k])
                    vmin = np.amin(W[:, :, l, k])
                    ax = fig.add_subplot(gs[gs_loc[l][0], gs_loc[l][1]])
                    self.image_plot(fig, ax, W[:, :, l, k],
                                    'Drill ' + pol_titles[self.UV.polarization_array[l]] +
                                    ' ' + flag_slice, vmin, vmax, aspect_ratio=aspect[plot_type], fraction=False,
                                    y_type='ant', x_type=x_type[plot_type])

                plt.tight_layout()
                fig.savefig(outpath + self.obs + '_Drill_' + flag_slice +
                            '_' + path_labels[plot_type] + str(uniques[k]) + '.png')
                plt.close(fig)

    def ant_pol_catalog(self, outpath, times, freqs):  # times and freqs should be of the same length

        def sigfig(x, s=4):  # s is number of sig-figs
            if x == 0:
                return(0)
            else:
                n = int(floor(log10(np.absolute(x))))
                y = 10**n * round(10**(-n) * x, s - 1)
                return(y)

        for time in times:
            for freq in freqs:

                fig, ax = plt.subplots(figsize=(14, 8))
                T = self.ant_pol_prepare(time, freq)
                title = self.obs + ' Ant-Pol Drill t = ' + str(time) + ' f = ' + \
                    str(sigfig(self.UV.freq_array[0, freq]) * 10**(-6)) + ' Mhz'
                vmax = np.amax(T)
                vmin = np.amin(T)

                self.image_plot(fig, ax, T, title, vmin, vmax, aspect_ratio=1, fraction=False,
                                y_type='ant-pol', x_type='ant-pol')

                plt.tight_layout()
                fig.savefig(outpath + self.obs + '_ant_pol_t' + str(time) +
                            '_f' + str(freq))
