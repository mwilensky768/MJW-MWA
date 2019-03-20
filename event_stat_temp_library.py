if events is not None and len(events):
          print('Beginning bl_avg flagging at %s' % time.strftime("%H:%M:%S"))
          for event in events:
              avg, counts, exp_counts, exp_error, bins = self._event_avg(data, event)
              lcut, rcut = self._cutoff(counts, bins, exp_counts, R_thresh)
              cut_cond = np.logical_or(avg > rcut, avg < lcut)
              cut_ind = np.where(cut_cond)
              temp_mask[event[0], cut_ind[0], 0, event[2]] = 1
              uv_grid = self._bl_grid(avg, event)
              for attr, calc in zip(attr_list, (avg, counts, exp_counts,
                                                exp_error, bins, uv_grid,
                                                np.array([lcut, rcut]))):
                  getattr(self, attr).append(calc)
          for attr in attr_list:
              setattr(self, attr, np.array(getattr(self, attr)))
          self.mask = temp_mask
          """The final output mask obtained from the calculations."""
          print('Done with bl_avg flagging at %s' % time.strftime("%H:%M:%S"))
      else:
          print('No events given to ES class. Not computing flags.')

def _event_avg(self, data, event):

    """
    This takes an event (time-frequency) and averages the visibility
    difference amplitudes across that event for each baseline. Then,
    an empirical thermal distribution is calculated using monte carlo,
    except when events are broad enough that the central limit theorem
    holds to good approximation (>30 frequency channels).

    Args:
        data: The sky-subtracted visibilities
        event: An event over which to average

    Returns:
        avg: The averaged data
        counts: The counts in each bin
        exp_counts: The expected counts from the thermal estimation
        exp_error: The expected variation of the counts in the bins (1
                   standard deviation)
        bins: The bin edges for the averaged amplitudes
    """

    avg = data[event[0], :, 0, event[2]]
    init_shape = avg.shape
    init_mask = avg.mask
    avg = avg.mean(axis=1)
    counts, bins = np.histogram(avg[np.logical_not(avg.mask)], bins='auto')
    sim_counts = np.zeros((self.MC_iter, len(counts)))
    # Simulate some averaged rayleigh data and histogram - take averages/variances of histograms
    for i in range(self.MC_iter):
        sim_data = np.random.rayleigh(size=init_shape,
                                      scale=np.sqrt(self.MLE[0, event[2]]))
        sim_data = sim_data.mean(axis=0)
        sim_counts[i, :], _ = np.histogram(sim_data, bins=bins)
    exp_counts = sim_counts.mean(axis=0)
    exp_error = np.sqrt(sim_counts.var(axis=0))

    return(avg, counts, exp_counts, exp_error, bins)

def _cutoff(self, counts, bins, exp_counts, R_thresh):

    """
    This function takes the histogrammed, averaged data, and compares it to
    the empirical distribution drawn from event_avg(). Cutoffs are drawn
    based on R_thresh, which is a ratio of counts.

    Args:
        counts: The counts in each bin
        bins: The bin edges for the averaged visibility difference amplitudes
        exp_counts: The expected counts in each bin
        R_thresh: The ratio of counts to exp_counts where the cutoff for
        outliers should be drawn
    Returns:
        lcut: The cutoff bin edge on the left-hand-side of the distribution
        rcut: The cutoff bin edge for the right-hand-side of the distribution
    """

    max_loc = bins[:-1][exp_counts.argmax()] + 0.5 * (bins[1] - bins[0])
    R = counts.astype(float) / exp_counts
    lcut_cond = np.logical_and(R > self.R_thresh, bins[1:] < max_loc)
    rcut_cond = np.logical_and(R > self.R_thresh, bins[:-1] > max_loc)
    if np.any(lcut_cond):
        lcut = bins[1:][max(np.where(lcut_cond)[0])]
    else:
        lcut = bins[0]
    if np.any(rcut_cond):
        rcut = bins[:-1][min(np.where(rcut_cond)[0])]
    else:
        rcut = bins[-1]

    return(lcut, rcut)

def _bl_grid(self, avg, event):

    """
    This takes time-frequency averaged data from event_avg() and coarsely
    grids it in the UV-plane at the time of the event. Each pixel is
    averaged across the baselines whose centers lie within the pixel.

    Args:
        avg: The averaged visibility difference amplitudes
        event: The event over which the average was performed
    Returns:
        uv_grid: A grid with average baseline power for baselines within each
        pixel, over the subband corresponding to the event.
    """

    u = self.uvw_array[event[0] * self.Nbls:(event[0] + 1) * self.Nbls, 0]
    v = self.uvw_array[event[0] * self.Nbls:(event[0] + 1) * self.Nbls, 1]
    uv_grid = np.zeros((len(self.pols), self.grid_dim, self.grid_dim))
    for i in range(self.grid_dim):
        for k in range(self.grid_dim):
            uv_grid[:, -k, i] = avg[np.logical_and(np.logical_and(u < self.grid[i + 1], self.grid[i] < u),
                                                   np.logical_and(v < self.grid[k + 1], self.grid[k] < v))].sum()

    return(uv_grid)


# Calculate MLE's with the INS flags in mind, and then apply choice of
# non-INS flags to the data
self.apply_flags(choice='INS', INS=self.INS)
VDH_kwargs = {'bins': bins,
              'fit_hist': fit_hist}
print('Preparing VDH at %s' % time.strftime("%H:%M:%S"))
self.VDH_prepare(**VDH_kwargs)
print('Done preparing VDH at %s ' % time.strftime("%H:%M:%S"))
self.apply_flags(choice=choice, custom=custom)
