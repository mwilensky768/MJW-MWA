import numpy as np
import pyuvdata as pyuv
from math import floor, ceil, log10, pi, log, sqrt
import os
import scipy.linalg
import SumThreshold as ST
import SIR as SIR
from scipy.special import erfinv
import rfiutil


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

    def one_d_hist_prepare(self, flag_slice='All', time_ind=slice(None),
                           bl_ind=slice(None), spw_ind=0,
                           freq_ind=slice(None), pol_ind=slice(None),
                           bins=None, fit='rayleigh', bin_window=[0, 1e+03],
                           writepath=''):
        """
        This function makes one_d visibility difference amplitude histograms.
        You may choose to histogram only subsets of the data using the set of ind
        keywords. You may pass indices or slice objects. Default is all the data
        in the 0th spectral window.

        You may give it a flag_slice. 'Unflagged' gives data not reported as
        contaminated. 'Flagged' gives data reported as contaminated.

        You may choose the bins by giving a sequence of bin edges, or 1000
        logarithmically spaced bins will be generated automatically.

        You may opt for a rayleigh superposition fit (by maximum likelihood
        estimation) by setting fit='rayleigh' and choosing an amplitude window
        over which to fit (bin_window). Give it bounds for data usage if there is
        suspected RFI.

        Set the writepath keyword for writing out function returns.
        """

        flags = self.flag_operations(flag_slice=flag_slice)[time_ind, bl_ind, spw_ind, freq_ind, pol_ind]
        values = np.absolute(self.UV.data_array)[time_ind, bl_ind, spw_ind, freq_ind, pol_ind]

        # Generate the bins or keep the selectred ones
        if bins is None:
            MIN = np.amin(values[values > 0])
            MAX = np.amax(values)
            bins = np.logspace(floor(log10(MIN)), ceil(log10(MAX)), num=1001)
        else:
            bins = bins

        # Generate a fit if desired
        if fit:
            m, fit = rfiutil.hist_fit(self.obs, bins, values, flags, writepath=writepath,
                                      flag_slice=flag_slice, bin_window=bin_window,
                                      fit_type=fit)
        else:
            m, bins = np.histogram(values[flags], bins=bins)
            fit = None

        # Write out the function returns
        base = '%s%s_%s_spw%i' % (writepath, self.obs, flag_slice, spw_ind)
        np.save('%s_hist.npy' % (base), m)
        np.save('%s_bins.npy' % (base), bins)
        np.save('%s_fit.npy' % (base), fit)

        return(m, bins, fit)

    def reverse_index(self, band, flag_slice='Unflagged'):
        # Find vis. differences within a certain amplitude band of the chosen flag slice

        flags = self.flag_operations(flag_slice=flag_slice)
        values = np.absolute(self.UV.data_array)

        ind = np.where((min(band) < values) & (values < max(band)) & (flags > 0))
        return(ind)

    def waterfall_hist_prepare(self, band, flag_slice='Unflagged', fraction=True,
                               writepath=''):
        # Generate a time-frequency histogram from reverse_index (sum over baselines)

        ind = self.reverse_index(band, flag_slice=flag_slice)
        H = np.zeros(self.UV.data_array.shape, dtype=int)
        H[ind] = 1
        H = H.sum(axis=1)
        if fraction:
            H = H.astype(float) / self.UV.Nbls
        np.save('%s%s_%s_whist.npy' % (writepath, self.obs, flag_slice),
                H)

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

    def INS_prepare(self, flag_slice='All', amp_avg='Amp', writepath=''):
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
        INS = np.absolute(np.mean(values, axis=1))
        frac_diff = INS / INS.mean(axis=0) - 1
        n, bins = np.histogram(frac_diff[np.logical_not(frac_diff.mask)], bins='auto')

        # This is how far out the data should extend if purely noisy
        bin_max = sqrt((pi - 4) / (pi * self.UV.Nbls) * 2 *
                       log(sqrt(pi) / (4 * len(INS[~INS.mask]) ** (2.0 / 3) * erfinv(0.5))))
        _, fit = rfiutil.hist_fit(self.obs, bins, frac_diff, np.logical_not(frac_diff.mask),
                                  bin_window=[-bin_max, bin_max], fit_type='normal')

        base = '%s%s_%s_%s' % (writepath, self.obs, flag_slice, amp_avg)
        np.ma.dump(INS, '%s_INS.npym' % (base))
        np.ma.dump(frac_diff, '%s_INS_frac_diff.npym' % (base))
        np.save('%s_INS_counts.npy' % (base), n)
        np.save('%s_INS_bins.npy' % (base), bins)
        np.save('%s_INS_fit.npy' % (base), fit)

        return(INS, frac_diff, n, bins, fit)

    def bl_scatter(self, mask):

        ind = np.where(mask)
        n, pol_counts = np.unique(np.vstack((ind[1], ind[3])), return_counts=True, axis=1)
        bl_avg = np.zeros([self.UV.Nbls, self.UV.Nspws, self.UV.Npols])
        ant_avg = np.zeros([self.UV.Nants_telescope, self.UV.Nspws, self.UV.Npols])
        for m in range(len(ind[0])):
            bl_avg[:, ind[1][m], ind[3][m]] += np.absolute(self.UV.data_array[ind[0][m], :, ind[1][m], ind[2][m], ind[3][m]])
        for m, pair in enumerate(n.transpose()):
            bl_avg[:, pair[0], pair[1]] = bl_avg[:, pair[0], pair[1]] / pol_counts[m]

        t0 = 18
        blt_slice = slice(self.UV.Nbls * t0, self.UV.Nbls * (t0 + 1))
        hist2d, xedges, yedges = np.histogram2d(self.UV.uvw_array[blt_slice, 0],
                                                self.UV.uvw_array[blt_slice, 1], bins=50)

        bl_hist = []
        bl_bins = []
        grid = np.zeros([self.UV.Nspws, self.UV.Npols, 50, 50])
        for m in range(self.UV.Nspws):
            hist, bins = np.histogram(bl_avg[:, m, :], bins='auto')
            bl_hist.append(hist)
            bl_bins.append(bins)
            for n in range(self.UV.Npols):
                for p in range(50):
                    for q in range(50):
                        seq = bl_avg[:, m, n][np.logical_and(np.logical_and(xedges[p] < self.UV.uvw_array[blt_slice, 0],
                                                                            self.UV.uvw_array[blt_slice, 0] < xedges[p + 1]),
                                                             np.logical_and(yedges[q] < self.UV.uvw_array[blt_slice, 1],
                                                                            self.UV.uvw_array[blt_slice, 1] < yedges[q + 1]))]
                        if len(seq) > 0:
                            grid[m, n, 49 - q, p] = np.mean(seq)

        return(bl_avg, bl_hist, bl_bins, hist2d, grid, xedges, yedges)
