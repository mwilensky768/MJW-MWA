import numpy as np
import pyuvdata as pyuv
from math import floor, ceil, log10, pi, log, sqrt
import os
import scipy.linalg
from scipy.special import erfinv
import rfiutil


class RFI:

    def __init__(self, obs, filepath, outpath, bad_time_indices=[0, -3, -2, -1],
                 filetype='uvfits', freq_chans=None, times=None, ant_str='cross',
                 polarizations=None):

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
                                     polarizations=polarizations)
        elif filetype is 'miriad':
            self.UV.read_miriad(filepath)
            self.UV.select(filepath, freq_chans=freq_chans, times=times,
                           polarizations=polarizations)
        if ant_str:
            self.UV.select(ant_str=ant_str)

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

        self.flag_titles = {'custom': 'Custom_Flag', 'COTTER': 'COTTER_Flags',
                            'INS': 'INS_Flags', None: 'All'}
        for item in self.flag_titles:
            if not os.path.exists('%sarrs/%s/' % (self.outpath, self.flag_titles[item])):
                os.makedirs('%sarrs/%s/' % (self.outpath, self.flag_titles[item]))
            assert(os.path.exists('%sarrs/%s/' % (self.outpath, self.flag_titles[item])))

    def apply_flags(self, choice=None, INS=None, custom=None):
        if choice is 'Original':
            self.UV.data_array.mask = self.UV.flag_array
        elif choice is 'INS':
            ind = np.where(INS.mask)
            self.UV.data_array[ind[0], :, ind[1], ind[2], ind[3]] = np.ma.masked
        elif choice is 'custom':
            self.UV.data_array[custom] = np.ma.masked
        elif np.any(self.UV.data_array.mask):
            self.UV.data_array.mask = False

    def one_d_hist_prepare(self, choice=None, INS=None, custom=None,
                           bins=None, fit=False):

        app_flags_kwargs = {'choice': choice,
                            'INS': INS,
                            'custom': custom}

        self.apply_flags(**app_flags_kwargs)

        # Generate the bins or keep the selected ones
        if bins is None:
            MIN = np.amin(self.UV.data_array)
            MAX = np.amax(self.UV.data_array)
            bins = np.logspace(np.floor(np.log10(MIN)), np.ceil(np.log10(MAX)), num=1001)

        if fit:
            bin_widths = np.diff(bins)
            bin_centers = bins[:-1] + 0.5 * bin_widths
            fit = np.zeros(len(bins) - 1)
            sig_arr, N_arr = np.sqrt(self.MLE_calc(axis=0, **app_flags_kwargs))
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

    def waterfall_hist_prepare(self, amp_range, choice=None, INS=None, custom=None,
                               fraction=True, axis=1):

        app_flags_kwargs = {'choice': choice,
                            'INS': INS,
                            'custom': custom}

        self.apply_flags(**app_flags_kwargs)

        # Find vis. diff. amps. in a given range. Could be unflagged data or all data.
        # and sum across a given axis to report the number of measurements affected
        # in the remaining space. Can also give as a fraction.
        H = np.logical_and(np.logical_and(min(amp_range) < self.UV.data_array,
                                          self.UV.data_array < max(amp_range)),
                           self.UV.data_array.mask == 0).sum(axis=axis)
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

    def INS(self, choice=None, INS=None, custom=None, sig_thresh=4,
            shape_dict={}, dt=1,
            match_filter=False, pow=1, typ='mean'):
        """
        Generate an incoherent noise spectrum.
        """

        app_flags_kwargs = {'choice': choice,
                            'INS': INS,
                            'custom': custom}

        match_filter_kwargs = {'sig_thresh': sig_thresh,
                               'shape_dict': shape_dict,
                               'dt': dt}

        self.apply_flags(**app_flags_kwargs)

        if np.any(self.UV.data_array.mask):
            Nbls = (np.logical_not(self.UV.data_array.mask)).sum(axis=1)
        else:
            Nbls = self.UV.Nbls * np.ones((self.UV.Ntimes - 1, ) +
                                          self.UV.data_array.shape[2:], dtype=int)

        INS = getattr(np, typ)(self.UV.data_array**pow, axis=1) / pow
        C = {'mean': [4 / np.pi - 1, 1],
             'var': [1, 1]}
        MS = (INS / INS.mean(axis=0) - 1) * np.sqrt(Nbls / C[typ][pow - 1])

        if match_filter:
            match_filter_args = (INS, MS, Nbls, self.outpath, self.UV.freq_array)
            INS, MS, events, hists = rfiutil.match_filter(*match_filter_args,
                                                          **match_filter_kwargs)
        else:
            events = None

        return(INS, MS, Nbls, events)

    def bl_grid_flag(self, INS_kwargs={}, MLE_kwargs={}, flag_kwargs={},
                     gridsize=50, edges=np.linspace(-3000, 3000, num=51)):

        INS, _, _, _, _, _, events = self.INS(**INS_kwargs)
        MLE_kwargs['flag_kwargs']['INS'] = INS
        MLE, _ = self.MLE_calc(**MLE_kwargs)
        self.apply_flags(**flag_kwargs)
        sim_data = np.ma.masked_array(np.random.rayleigh(size=(self.UV.data_array.shape),
                                                         scale=np.sqrt(MLE)))
        sim_data[self.UV.data_array.mask] = np.ma.masked
        temp_mask = np.zeros(self.UV.data_array.shape, dtype=bool)
        temp_mask[self.UV.data_array.mask] = 1

        Nevent = len(events)
        bl_avg = np.ma.masked_array(np.zeros(self.UV.Nbls))
        grid = np.ma.masked_array(np.zeros([gridsize, gridsize, Nevent]))
        bl_hists, bl_bins, sim_hists, cutoffs, max_locs, flags = [], [], [], [], [], []

        for m in range(Nevent):
            bl_avg = self.UV.data_array[events[m, 3], :, events[m, 0], events[m, 2], events[m, 1]].mean(axis=1)
            sim_avg = sim_data[events[m, 3], :, events[m, 0], events[m, 2], events[m, 1]].mean(axis=1)

            blt_slice = slice(self.UV.Nbls * events[m, 3],
                              self.UV.Nbls * (events[m, 3] + 1))

            hist, bins = np.histogram(bl_avg[np.logical_not(bl_avg.mask)], bins='auto')
            sim_hist, _ = np.histogram(sim_avg[np.logical_not(sim_avg.mask)], bins=bins)

            max_loc = bins[:-1][sim_hist.argmax()] + 0.5 * (bins[1] - bins[0])
            R = hist.astype(float) / sim_hist.astype(float)
            lcut_cond = np.logical_and(R > 10, bins[1:] < max_loc)
            rcut_cond = np.logical_and(R > 10, bins[:-1] > max_loc)
            if np.any(lcut_cond):
                lcut = bins[1:][max(np.where(lcut_cond)[0])]
            else:
                lcut = bins[0]
            if np.any(rcut_cond):
                rcut = bins[:-1][min(np.where(rcut_cond)[0])]
            else:
                rcut = bins[-1]
            cut_cond = np.logical_or(bl_avg > rcut, bl_avg < lcut)
            np.save('%sarrs/Flag_Fraction_%i.npy' % (self.outpath, m), cut_cond.mean())
            if np.any(cut_cond):
                temp_mask[events[m, 3], cut_cond, events[m, 0], events[m, 2], events[m, 1]] = 1

            bl_hists.append(hist)
            sim_hists.append(sim_hist)
            cutoffs.append((lcut, rcut))
            bl_bins.append(bins)
            max_locs.append(max_loc)

            for i in range(50):
                for k in range(50):
                    seq = bl_avg[np.logical_and(np.logical_and(edges[i] < self.UV.uvw_array[blt_slice, 0],
                                                               self.UV.uvw_array[blt_slice, 0] < edges[i + 1]),
                                                np.logical_and(edges[k] < self.UV.uvw_array[blt_slice, 1],
                                                               self.UV.uvw_array[blt_slice, 1] < edges[k + 1]))]
                    if len(seq) > 0:
                        grid[49 - k, i, m] = np.mean(seq)

        self.UV.data_array.mask = temp_mask
        np.save('%sarrs/Mean_Flag_Per_Time.npy' % (self.outpath), temp_mask.mean(axis=(1, 2, 3, 4)))
        np.save('%sarrs/Mean_Flag.npy' % (self.outpath), temp_mask.mean())
        INS = self.UV.data_array.mean(axis=1)
        MS = INS / INS.mean(axis=0) - 1

        return(grid, bl_bins, bl_hists, sim_hists, cutoffs, events, INS, MS)

    def MLE_calc(self, axis=None, choice=None, INS=None, custom=None):
        # Calculate the rms of the unflagged data, splitting according to axis tuple

        app_flags_kwargs = {'choice': choice,
                            'INS': INS,
                            'custom': custom}

        self.apply_flags(**app_flags_kwargs)
        MLE = 0.5 * np.mean(self.UV.data_array**2, axis=axis)
        if np.any(self.UV.data_array.mask):
            N = np.count_nonzero(np.logical_not(self.UV.data_array.mask), axis=axis)
        else:
            N = self.UV.Nblts

        fn = filter(str.isalnum, str(axis))

        base = '%sarrs/%s/%s_' % (self.outpath, self.flag_titles[flag_kwargs['choice']], self.obs)
        np.ma.dump(MLE, '%s_MLE_%s.npym' % (base, fn))
        np.save('%s_N_%s.npy' % (base, fn), N)

        return(MLE, N)
