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

    def __init__(self, obs, filepath, outpath, bad_time_indices=[0, -3, -2, -1],
                 filetype='uvfits', freq_chans=None, times=None, ant_str='cross'):

        # These lines establish the most basic attributes of the class, namely
        # its base UVData object and the obsid
        self.obs = obs
        self.UV = pyuv.UVData()
        self.outpath = outpath
        if filetype is 'uvfits':
            if bad_time_indices:
                self.UV.read_uvfits(filepath, read_data=False)
                times = np.unique(self.UV.time_array).tolist()
                bad_times = []
                for k in bad_time_indices:
                    bad_times.append(times[k])
                for bad_time in bad_times:
                    times.remove(bad_time)
            self.UV.read_uvfits_data(filepath, freq_chans=freq_chans, times=times,
                                     ant_str=ant_str)
        elif filetype is 'miriad':
            self.UV.read_miriad(filepath)

        # This generalizes polarization references during plotting
        pol_keys = range(-8, 5)
        pol_keys.remove(0)
        pol_values = ['YX', 'XY', 'YY', 'XX', 'LR', 'RL', 'LL', 'RR', 'I', 'Q',
                      'U', 'V']
        pol_dict = dict(zip(pol_keys, pol_values))
        self.pols = [pol_dict[self.UV.polarization_array[k]] for k in
                     range(self.UV.Npols)]

        # These ensure that every baseline reports at every time so that subtraction
        # can go off without a hitch
        assert self.UV.Nblts == self.UV.Nbls * self.UV.Ntimes, 'Nblts != Nbls * Ntimes'
        cond = np.all([self.UV.baseline_array[:self.UV.Nbls] ==
                       self.UV.baseline_array[k * self.UV.Nbls:(k + 1) * self.UV.Nbls]
                       for k in range(1, self.UV.Ntimes - 1)])
        assert cond, 'Baseline array slices do not match!'

        # Make data array, flag array, and outpaths

        self.UV.data_array = np.ma.masked_array(np.absolute(np.diff(np.reshape(self.UV.data_array,
                                                [self.UV.Ntimes, self.UV.Nbls, self.UV.Nspws,
                                                 self.UV.Nfreqs, self.UV.Npols]), axis=0)))

        self.UV.flag_array = np.reshape((self.UV.flag_array[:-self.UV.Nbls] +
                                         self.UV.flag_array[self.UV.Nbls:]) > 0,
                                        [self.UV.Ntimes - 1, self.UV.Nbls,
                                         self.UV.Nspws, self.UV.Nfreqs,
                                         self.UV.Npols]).astype(bool)

        self.flag_titles = {False: 'All', True: 'Post_Flag'}
        for item in self.flag_titles:
            if not os.path.exists('%sarrs/%s/' % (self.outpath, self.flag_titles[item])):
                os.makedirs('%sarrs/%s/' % (self.outpath, self.flag_titles[item]))

    def apply_flags(self, app=False):
        if app:
            if ~np.all(self.UV.data_array.mask == self.UV.flag_array):
                self.UV.data_array.mask = self.UV.flag_array
        elif np.any(self.UV.data_array.mask):
            self.UV.data_array.mask = False

    def one_d_hist_prepare(self, flag=False, bins=None, fit=False,
                           bin_window=None):
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
        """
        self.apply_flags(flag)

        # Generate the bins or keep the selected ones
        if bins is None:
            MIN = np.amin(self.UV.data_array)
            MAX = np.amax(self.UV.data_array)
            bins = np.logspace(floor(log10(MIN)), ceil(log10(MAX)), num=1001)

        if fit:
            bin_widths = np.diff(bins)
            bin_centers = bins[:-1] + 0.5 * bin_widths
            fit = np.zeros(len(bins) - 1)
            if bin_window:
                self.UV.data_array = np.ma.masked_outside(self.UV.data_array,
                                                          min(bin_window),
                                                          max(bin_window))
            sig_arr, N_arr = self.rms_calc(axis=(0, 1)) / np.sqrt(2)
            if bin_window:
                self.apply_flags(flag)
            for p in range(sig_arr.shape[0]):
                for q in range(sig_arr.shape[1]):
                    for r in range(sig_arr.shape[2]):
                        if N_arr[p, q, r] > 0:
                            fit += N_arr[p, q, r] * bin_widths * (bin_centers / sig_arr[p, q, r]**2) * \
                                np.exp(- bin_centers**2 / (2 * sig_arr[p, q, r]**2))
        else:
            fit = None

        n, _ = np.histogram(self.UV.data_array[np.logical_not(self.UV.data_array.mask)], bins=bins)

        # Write out the function returns
        base = '%sarrs/%s/%s' % (self.outpath, self.flag_titles[flag], self.obs)
        np.save('%s_hist.npy' % (base), n)
        np.save('%s_bins.npy' % (base), bins)
        np.save('%s_fit.npy' % (base), fit)

        return(n, bins, fit)

    def waterfall_hist_prepare(self, amp_range, flag=False, fraction=True, axis=1):

        self.apply_flags(flag)

        # Find vis. diff. amps. in a given range. Could be unflagged data or all data.
        # and sum across a given axis to report the number of measurements affected
        # in the remaining space. Can also give as a fraction.
        H = ((min(amp_range) < self.UV.data_array) &
             (self.UV.data_array < max(amp_range)) &
             (self.UV.data_array.mask == 0)).sum(axis=axis)
        if fraction:
            # ugly if statement for syntactical reasons...
            if type(axis) is int:
                H = H.astype(float) / self.UV.data_array.shape[axis]
            else:
                H = H.astype(float) / np.prod([self.UV.data_array.shape[k] for k in axis])

        np.ma.dump(H, '%sarrs/%s/%s_whist.npym' % (self.outpath,
                                                   self.flag_titles[flag],
                                                   self.obs))

        return(H)

    def ant_pol_prepare(self, time, freq, spw, amp=False):

        """
        This function no longer works since I am only using amplitudes right now
        """

        dim = 2 * self.UV.Nants_telescope

        T = np.zeros([dim, dim])
        q_keys = [self.pols[k] for k in range(self.UV.Npols)]
        q_values = [[0, 0], [self.UV.Nants_telescope, self.UV.Nants_telescope],
                    [0, self.UV.Nants_telescope], [self.UV.Nants_telescope, 0]]

        q = dict(zip(q_keys, q_values))

        for m in range(self.UV.Nbls):
            for n in range(self.UV.Npols):
                A = self.UV.data_array[time, m, spw, freq, n]
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

    def INS_prepare(self, flag=False, sig_thresh=4):
        """
        Generate an incoherent noise spectrum. The amp_avg keyword determines
        the order of amplitude vs. average.
        """

        self.apply_flags(flag)

        Nbls_arr = (~self.UV.data_array.mask).sum(axis=1)
        INS = np.mean(self.UV.data_array, axis=1)
        MS = INS / INS.mean(axis=0) - 1
        n, bins = np.histogram(MS[~MS.mask], bins='auto')

        fit = rfiutil.INS_hist_fit(bins, MS[~MS.mask], Nbls_arr, sig_thresh)

        base = '%sarrs/%s/%s' % (self.outpath, self.flag_titles[flag], self.obs)
        np.ma.dump(INS, '%s_INS.npym' % (base))
        np.ma.dump(MS, '%s_INS_MS.npym' % (base))
        np.save('%s_INS_Nbls.npy' % (base), Nbls_arr)
        np.save('%s_INS_counts.npy' % (base), n)
        np.save('%s_INS_bins.npy' % (base), bins)
        np.save('%s_INS_fit.npy' % (base), fit)

        return(INS, MS, Nbls_arr, n, bins, fit)

    def bl_grid(self, mask):
        ind, event_bound = rfiutil.event_identify(mask)
        Nevent = len(event_bound) + 1
        bl_avg = np.zeros([self.UV.Nbls, Nevent])
        # Average vis. diff. amps. for each event
        bl_hist2d = []
        bl_hist = []
        bl_bins = []
        grid = np.zeros([50, 50, Nevents])
        edges = np.linspace(-3000, 3000, num=51)

        for m in range(Nevent):
            # Set up the appropriate slices
            # Event bounds tell us the last index of an event
            # Start at 0 or the last event bound + 1
            # End at event bound (slice syntax -> event_bound[m] + 1), or the end
            if m == 0:
                p = 0
            else:
                p = event_bound[m - 1] + 1
            if m < Nevent - 1:
                q = event_bound[m] + 1
            else:
                q = len(ind[0])
            event_slice = slice(p, q)
            bl_avg[:, m] = np.absolute(self.UV.data_array[ind[2][event_slice],
                                                          :, ind[0][event_slice],
                                                          ind[3][event_slice],
                                                          ind[1][event_slice]]).mean(axis=0)

            blt_slice = slice(self.UV.Nbls * ind[2][p], self.UV.Nbls * (ind[2][p] + 1))
            hist2d, xedges, yedges = np.histogram2d(self.UV.uvw_array[blt_slice, 0],
                                                    self.UV.uvw_array[blt_slice, 1],
                                                    bins=edges)
            bl_hist2d.append(hist2d)
            hist, bins = np.histogram(bl_avg[:, m], bins='auto')
            bl_hist.append(hist)
            bl_bins.append(bins)

            for i in range(50):
                for k in range(50):
                    seq = bl_avg[:, m][np.logical_and(np.logical_and(xedges[i] < self.UV.uvw_array[blt_slice, 0],
                                                                     self.UV.uvw_array[blt_slice, 0] < xedges[i + 1]),
                                                      np.logical_and(yedges[k] < self.UV.uvw_array[blt_slice, 1],
                                                                     self.UV.uvw_array[blt_slice, 1] < yedges[k + 1]))]
                    if len(seq) > 0:
                        grid[49 - k, i, m] = np.mean(seq)

        return(bl_avg, ind, event_bound, bl_hist, bl_bins, bl_hist2d, grid, edges)

    def ant_grid(self, mask):

        bl_avg, _, _, _, _, _, _, _ = self.bl_scatter(mask)
        counts = np.zeros(self.UV.Nants_telescope)
        Nevents = len(event_bound) + 1
        ant_avg = np.zeros([self.UV.Nants_telescope, Nevents])
        for m in range(Nevents):
            counts[m] = len(np.where(self.UV.ant_1_array == m)[0]) + \
                len(np.where(self.UV.ant_2_array == m)[0])
        for m in range(self.UV.Nspws):
            for n in range(self.UV.Npols):
                for p in range(self.UV.Nbls):
                    ant_avg[ant_1_array[p], m, n] += bl_avg[p, m, n]
                    ant_avg[ant_2_array[p], m, n] += bl_avg[p, m, n]
                ant_avg[:, m, n] = ant_avg[:, m, n] / counts

        _, xedges, yedges = np.histogram2d(self.UV.antenna_positions[:, 0],
                                           self.UV.antenna_positions[:, 1],
                                           bins=10)

        ant_grid = np.zeros([self.UV.Nspws, self.UV.Npols, 10, 10])
        for m in range(self.UV.Nspws):
            for n in range(self.UV.Npols):
                for p in range(10):
                    for q in range(10):
                        seq = ant_avg[:, m, n][np.logical_and(np.logical_and(xedges[p] < self.UV.antenna_positions[:, 0],
                                                                             self.UV.antenna_positions[:, 0] < xedges[p + 1]),
                                                              np.logical_and(yedges[q] < self.UV.antenna_positions[:, 1],
                                                                             self.UV.antenna_positions[:, 1] < yedges[q + 1]))]
                        if len(seq) > 0:
                            ant_grid[m, n, 9 - q, p] = np.mean(seq)

        return(ant_avg, ant_grid, xedges, yedges)

    def rms_calc(self, axis=None, flag=True):
        # Calculate the rms of the unflagged data, splitting according to axis tuple
        self.apply_flags(flag)
        rms = np.sqrt(np.mean(self.UV.data_array**2, axis=axis))
        N = np.count_nonzero(np.logical_not(self.UV.data_array.mask), axis=axis)

        fn = filter(str.isalnum, str(axis))

        base = '%sarrs/%s/%s_' % (self.outpath, self.flag_titles[flag], self.obs)
        np.ma.dump(rms, '%s_rms_%s.npym' % (base, fn))
        np.save('%s_N_%s.npy' % (base, fn), N)

        return(rms, N)
