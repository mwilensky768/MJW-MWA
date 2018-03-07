import numpy as np
import pyuvdata as pyuv
from math import floor, ceil, log10, pi
import os
import scipy.linalg


class RFI:

    def __init__(self, obs, filepath, bad_time_indices=[0, -3, -2, -1], coarse_band_remove=False,
                 auto_remove=True, filetype='uvfits', good_freq_indices=[]):

        # These lines establish the most basic attributes of the class, namely
        # its base UVData object and the obsid
        self.obs = obs
        self.UV = pyuv.UVData()
        if filetype is 'uvfits':
            self.UV.read_uvfits(filepath)
        elif filetype is 'miriad':
            self.UV.read_miriad(filepath)

        # This generalizes polarization references during plotting
        pol_keys = [-8 + k for k in range(13)]
        pol_keys.remove(0)
        pol_values = ['YX', 'XY', 'YY', 'XX', 'LR', 'RL', 'LL', 'RR', 'I', 'Q',
                      'U', 'V']
        pol_dict = dict(zip(pol_keys, pol_values))
        self.pols = [pol_dict[self.UV.polarization_array[k]] for k in
                     range(self.UV.Npols)]

        # These if conditionals check for keywords which down-select the UVData
        # object
        if bad_time_indices:
            times = np.unique(self.UV.time_array).tolist()
            bad_times = []
            for k in bad_time_indices:
                bad_times.append(times[k])
            for bad_time in bad_times:
                times.remove(bad_time)
            self.UV.select(times=times)

        if good_freq_indices:
            self.UV.select(freq_chans=good_freq_indices)

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

        # These ensure that every baseline reports at every time so that subtraction
        # can go off without a hitch
        assert self.UV.Nblts == self.UV.Nbls * self.UV.Ntimes, 'Nblts != Nbls * Ntimes'
        cond = np.all([self.UV.baseline_array[:self.UV.Nbls] ==
                       self.UV.baseline_array[k * self.UV.Nbls:(k + 1) * self.UV.Nbls]
                       for k in range(1, self.UV.Ntimes - 1)])
        assert cond, 'Baseline array slices do not match!'
        self.UV.data_array = np.diff(np.reshape(self.UV.data_array,
                                     [self.UV.Ntimes, self.UV.Nbls, self.UV.Nspws,
                                      self.UV.Nfreqs, self.UV.Npols]), axis=0)

    def flag_operations(self, flag_slice='Unflagged'):
        # This function logically combines the flags for subtracted visibilities

        A = np.reshape(self.UV.flag_array, [self.UV.Ntimes, self.UV.Nbls, self.UV.Nspws,
                       self.UV.Nfreqs, self.UV.Npols])

        # Neither visibility was flagged
        if flag_slice is 'Unflagged':
            A = np.logical_not(np.logical_or(A[0:(self.UV.Ntimes - 1), :, :, :, :],
                                             A[1:self.UV.Ntimes, :, :, :, :]))
        # Either visibility was flagged, or both
        elif flag_slice is 'Flagged':
            A = np.logical_or(A[0:(self.UV.Ntimes - 1), :, :, :, :], A[1:self.UV.Ntimes, :, :, :, :])
        # Both visibilities were flagged
        elif flag_slice is 'And':
            A = np.logical_and(A[0:(self.UV.Ntimes - 1), :, :, :, :], A[1:self.UV.Ntimes, :, :, :, :])
        # One or the other visibilit was flagged, not both
        elif flag_slice is 'XOR':
            A = np.logical_xor(A[0:(self.UV.Ntimes - 1), :, :, :, :], A[1:self.UV.Ntimes, :, :, :, :])
        # Ignore the flags. Work with all of the data.
        elif flag_slice is 'All':
            A = np.ones([self.UV.Ntimes - 1, self.UV.Nbls, self.UV.Nspws,
                         self.UV.Nfreqs, self.UV.Npols], dtype=bool)

        return(A)

    def rayleigh_fit(bins, values, flags, write=False, writepath='',
                     flag_slice='All', bin_window=[0, 1e3], fit_type='rayleigh'):

        bin_widths = np.diff(bins)
        bin_centers = bins[:-1] + 0.5 * bin_widths
        fit = np.zeros(len(bins) - 1)

        if fit_type is 'rayleigh':
            m = np.copy(fit)
            if write:
                sigma_array = np.zeros(values.shape[3])
            # Calculate maximum likelihood estimator for each frequency and polarization
            for l in range(values.shape[4]):
                for k in range(values.shape[3]):
                    n, bins = np.histogram(values[:, :, :, k, l][flags[:, :, :, k, l]], bins=bins)
                    N = np.sum(n)
                    m += n
                    # Only use data within the fit window
                    data_cond = np.logical_and(np.logical_and(min(bin_window) < values[:, :, :, k, l],
                                               values[:, :, :, k, l] < max(bin_window)), flags[:, :, :, k, l])
                    N_fit = np.count_nonzero(data_cond)
                    # Do not calculate a sigma if nothing belonging to the flag slice is in the window
                    if N_fit > 0:
                        # Calculate the MLE and histogram according to the MLE
                        sigma = np.sqrt(0.5 * np.sum(values[:, :, :, k, l][data_cond]**2) / N_fit)
                        fit += N_fit * bin_widths * (1 / sigma**2) * bin_centers * \
                            np.exp(-bin_centers**2 / (2 * sigma ** 2))
                        if write:
                            sigma_array[k] = sigma
                    elif write:
                        sigma = 0
                        sigma_array[k] = sigma
                if write:
                    np.save('%s%s_%s_sigma_%s.npy' % (writepath, self.obs,
                                                      flag_slice,
                                                      self.pols[l]),
                            sigma_array)
        if fit_type is 'normal':
            data_cond = np.logical_and(np.logical_and(min(bin_window) < values,
                                                      values < max(bin_window)),
                                       flags)
            N = len(values[data_cond])
            mu = np.mean(values[data_cond])
            sigma_sq = np.var(values[data_cond])
            fit = N * bin_widths / np.sqrt(2 * pi * sigma_sq) * \
                np.exp(-((bin_centers - mu) ** 2) / sigma_sq)

        return(m, fit)

    def one_d_hist_prepare(self, flag_slice='Unflagged', time_drill=None,
                           freq_drill=None, freq_exc=None, time_exc=None,
                           bins=None, fit='rayleigh', bin_window=[0, 1e+03],
                           write=False, writepath='', label='', pol_drill=None):
        """
        This function makes one_d visibility difference amplitude histograms.
        You may choose to make histograms for a single time or frequency using
        time_drill and freq_drill respectively, or you may exclude a single time
        or frequency using time_exc or freq_exc respectively.

        You may choose the bins by giving a sequence of bin edges, or 1000
        logarithmically spaced bins will be generated automatically.

        You may opt for a rayleigh superposition fit (by maximum likelihood
        estimation) by setting fit=True and choosing an amplitude window over
        which to fit. Give it bounds.

        You may opt to write out the function returns, in which case set the
        write and writepath keywords appropriately.

        You may give it a label, for plot legends.
        """

        flags = self.flag_operations(flag_slice=flag_slice)
        values = np.absolute(self.UV.data_array)

        # Drill or exclude specific times/frequencies
        if time_drill:
            values = values[time_drill:time_drill + 1, :, :, :, :]
            flags = flags[time_drill:time_drill + 1, :, :, :, :]
        if time_exc:
            values = np.concatenate((values[:time_exc, :, :, :, :],
                                     values[time_exc + 1:, :, :, :, :]), axis=0)
            flags = np.concatenate((flags[:time_exc, :, :, :, :],
                                    flags[time_exc + 1:, :, :, :, :]), axis=0)
        if freq_drill:
            values = values[:, :, :, freq_drill:freq_drill + 1, :]
            flags = flags[:, :, :, freq_drill:freq_drill + 1, :]
        if freq_exc:
            values = np.concatenate((values[:, :, :, :freq_exc, :],
                                     values[:, :, :, freq_exc + 1:, :]), axis=3)
            flags = np.concatenate((flags[:, :, :, :freq_exc, :],
                                    flags[:, :, :, freq_exc + 1:, :]), axis=3)
        if pol_drill:
            values = values[:, :, :, :, pol_drill:pol_drill + 1]
            flags = flags[:, :, :, :, pol_drill:pol_drill + 1]

        # Generate the bins or keep the selectred ones
        if bins is None:
            MIN = np.amin(values[values > 0])
            MAX = np.amax(values)
            bins = np.logspace(floor(log10(MIN)), ceil(log10(MAX)), num=1001)
        else:
            bins = bins

        # Generate a fit if desired
        if fit:
            m, bins = hist_fit(bins, values, flags, fit_type=fit, write=write,
                               writepath=writepath, flag_slice=flag_slice,
                               bin_window=bin_window)
        else:
            m, bins = np.histogram(values[flags], bins=bins)
            fit = None

        # Write out the function returns
        if write:
            np.save('%s%s_%s_hist.npy' % (writepath, self.obs, flag_slice), m)
            np.save('%s%s_%s_bins.npy' % (writepath, self.obs, flag_slice), bins)
            np.save('%s%s_%s_fit.npy' % (writepath, self.obs, flag_slice), fit)

        return(m, bins, fit, label)

    def reverse_index(self, band, flag_slice='Unflagged'):
        # Find vis. differences within a certain amplitude band of the chosen flag slice

        flags = self.flag_operations(flag_slice=flag_slice)
        values = np.absolute(self.UV.data_array)

        ind = np.where((min(band) < values) & (values < max(band)) & (flags > 0))
        return(ind)

    def waterfall_hist_prepare(self, band, flag_slice='Unflagged', fraction=True):
        # Generate a time-frequency histogram from reverse_index (sum over baselines)

        ind = self.reverse_index(band, flag_slice=flag_slice)
        H = np.zeros(self.UV.data_array.shape, dtype=int)
        H[ind] = 1
        H = np.sum(H, axis=1)
        if fraction:
            H = H.astype(float) / self.UV.Nbls

        return(H)

    def drill_hist_prepare(self, band, flag_slice='All', drill_type='time'):
        # Generate antenna-time or antenna-frequency reverse index histograms

        ind = self.reverse_index(band, flag_slice=flag_slice)

        # Map baselines to antenna indices
        ant1_ind, ant2_ind = [], []
        for inds in ind[1]:
            ant1_ind.append(self.UV.ant_1_array[inds])
            ant2_ind.append(self.UV.ant_2_array[inds])
        ant_ind = [np.array(ant1_ind), np.array(ant2_ind)]

        # Accrue counts
        if drill_type is 'time':
            uniques = np.unique(ind[0])
            H = np.zeros([self.UV.Nants_telescope, self.UV.Nfreqs, self.UV.Npols,
                          self.UV.Ntimes - 1])
            for p in range(2):
                H[(ant_ind[p], ind[3], ind[4], ind[0])] += 1
        elif drill_type is 'freq':
            uniques = np.unique(ind[3])
            H = np.zeros([self.UV.Nants_telescope, self.UV.Ntimes - 1, self.UV.Npols,
                          self.UV.Nfreqs])
            for p in range(2):
                H[(ant_ind[p], ind[0], ind[4], ind[3])] += 1

        return(H, uniques)

    def ant_pol_prepare(self, time, freq, amp=True):

        """
        Generate an array for a time and frequency divided into quadrants.
        Each quadrant belongs to a polarization. [1, 2, 3, 4] -> [XY, XX, YX, YY].
        Each quadrant is divided in half. The top right has the imag. component
        of the vis. diff. while the bottom left has the real comp. of the vis. dif.
        You may choose to calculate the amplitude of the real or imag. component,
        eliminating some phase information in favor of a one-sided colorbar by
        setting amp=True.
        """

        dim = 2 * self.UV.Nants_telescope

        T = np.zeros([dim, dim])
        q_keys = [self.pols[k] for k in range(self.UV.Npols)]
        q_values = [[0, 0], [self.UV.Nants_telescope, self.UV.Nants_telescope],
                    [0, self.UV.Nants_telescope], [self.UV.Nants_telescope, 0]]

        q = dict(zip(q_keys, q_values))

        for m in range(self.UV.Nbls):
            for n in range(self.UV.Npols):
                A = self.UV.data_array[time, m, 0, freq, n]
                if amp:
                    T[self.UV.ant_1_array[m] + q[self.pols[n]][0],
                      self.UV.ant_2_array[m] + q[self.pols[n]][1]] = np.absolute(A.imag)
                    T[self.UV.ant_2_array[m] + q[self.pols[n]][0],
                      self.UV.ant_1_array[m] + q[self.pols[n]][1]] = np.absolute(A.real)
                else:
                    T[self.UV.ant_1_array[m] + q[self.pols[n]][0],
                      self.UV.ant_2_array[m] + q[self.pols[n]][1]] = A.imag
                    T[self.UV.ant_2_array[m] + q[self.pols[n]][0],
                      self.UV.ant_1_array[m] + q[self.pols[n]][1]] = A.real

        return(T)

    def INS_prepare(self, flag_slice='All', amp_avg='Amp', write=False,
                    writepath=''):
        """
        Generate an incoherent noise spectrum. The amp_avg keyword determines
        the order of amplitude vs. average.
        """

        if amp_avg is 'Amp':
            values = np.absolute(self.UV.data_array)
        elif amp_avg is 'Avg':
            values = self.UV.data_array

        flags = self.flag_operations(flag_slice)
        values = np.ma.masked_array(values)
        values[np.logical_not(flags)] = np.ma.masked
        avg = np.absolute(np.mean(values, axis=1))
        frac_diff = avg / np.mean(avg, axis=0) - 1
        n, bins = np.histogram(frac_diff.flatten())
        sig = np.sqrt((4 - pi) / (pi * self.UV.Nbls))
        _, fit = self.hist_fit(bins, frac_diff, np.logical_not(frac_diff.mask),
                               bin_window=[-4 * sig, 4 * sig], fit_type='normal')

        if write:
            np.save('%s%s_Vis_Avg_%s_%s.npy' %
                    (writepath, self.obs, amp_avg, flag_slice), avg)

        return(avg, frac_diff, n, bins, fit)

    def ant_scatter_prepare(self):

        # Do the least squares fit
        A = np.c_[self.UV.antenna_positions[:, 0],
                  self.UV.antenna_positions[:, 1],
                  np.ones(len(self.UV.Nants_telescope))]
        C, _, _, _ = scipy.linalg.lstsq(A, self.UV.antenna_positions[:, 3])

        # Construct the normal vector to the plane and normalize it
        n = np.array([-C[0], -C[1], 1])
        n = n / np.sqrt(np.sum(n * n))

        # Construct original basis vectors
        x = np.array([1, 0, 0])
        y = np.array([0, 1, 0])

        # Perform Graham-Schmidt
        u = x - np.sum(n * x) * n
        u = u / np.sqrt(np.sum(u * u))

        v = y - np.sum(n * y) * n - np.sum(u * y) * u
        v = v / np.sqrt(np.sum(v * v))

        # Construct transformation matrix
        B = np.c_[u, v, n]

        # Transform the antenna locations
        ant_locs = np.transpose(self.UV.antenna_positions)
        for k in range(len(self.UV.Nants_telescope)):
            ant_locs[:, k] = np.matmult(B, ant_locs[:, k])
        ant_locs = np.transpose(ant_locs)[:, :2]

        return(ant_locs)
