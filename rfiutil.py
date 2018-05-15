import numpy as np
from math import floor, ceil, log10, pi, log, sqrt
from scipy.special import erfinv


def SumThreshold(x, y, M, chi):

    #  x (float array) is the data itself (1D time at time series at given frequency or vice versa)
    #  y (binary [coded as int] array) is the previous flag mask (as a NUMPY ARRAY) - all zeros if no flagging has been done
    # M (int) is the desired subsequence
    # chi (float) is the threshold (regarded as a function of M in the paper)

    N = len(x)

    #  These are for use in the loops, t will be the new flag mask
    z = 0
    q = 0
    count = 0
    t = np.copy(y)

    # This loop creates the window
    while q < M:
        if not y[q]:
            z += x[q]
            count += 1
        q += 1

    # This loop slides the window
    while q < N:
        if abs(z) > count * chi:
            t[q - M:q] = np.ones(M, bool)  # Flag subsequence of length M if exceeds threshold
        if not y[q]:  # add x[q] to subsequence
            z += x[q]
            count += 1
        if not y[q - M]:  # take x[q-M] out of subsequence
            z -= x[q - M]
            count -= 1
        q += 1  # shift window

    return(t.astype(bool))


def SIROperator(FMi, Agg):  # Takes a (1-d) flag mask and aggression param. (scalar) as input

    N = len(FMi)

    Psi = FMi.astype(float) + Agg - 1  # Initialize psi (this is a temp. array)
    M = np.zeros(N, float)  # Initialize M (this is also temp.)

    for n in range(N - 1):  # Define M as in AOFlagger paper
        M[n + 1] = M[n] + Psi[n]

    P = np.zeros(N, int)  # Initialize P - this is a temp. array which is to be constructed so that M(P(n)) = min M(i), 0 <= i <= n (perhaps to be called the "latest min index")

    for n in range(1, N):  # This loop is really clever - I probably wouldn't have come up with it
        P[n] = P[n - 1]  # RHS is the latest minimum
        if M[P[n]] > M[n]:  # Satisfaction of this is to say a new latest min has been found
            P[n] = n

    Q = np.zeros(N, int)  # Similar to P, but looks for max M(j) x <= j <= N-1 (perhaps "earliest max index")
    Q[N - 1] = N - 1

    for n in range(1, N):  # Similar loop as before - but has to count backwards
        Q[N - 1 - n] = Q[N - n]
        if M[Q[N - 1 - n]] < M[N - n]:
            Q[N - 1 - n] = N - n

    FMf = np.zeros(N, int)  # Initialize output flag mask

    for n in range(N):  # Ask important flagging question
        if M[Q[n]] - M[P[n]] >= 0:
            FMf[n] = 1
        else:
            FMf[n] = 0

    return(FMf)


def sigma_calc(Nbls):
    sigma = np.sqrt((4 - pi) / (Nbls * pi))

    return(sigma)


def INS_hist_fit(bins, MS, Nbls, sig_thresh):

    bin_widths = np.diff(bins)
    bin_centers = bins[:-1] + 0.5 * bin_widths

    sigma = sigma_calc(np.amax(Nbls))
    thresh = sig_thresh * sigma
    data_cond = np.logical_and(-thresh < MS, MS < thresh)

    N = len(MS[data_cond])
    mu = np.mean(MS[data_cond])
    sigma_sq = np.var(MS[data_cond])
    fit = N * bin_widths / np.sqrt(2 * pi * sigma_sq) * \
        np.exp(-((bin_centers - mu) ** 2) / (2 * sigma_sq))

    return(fit)


def edge_detect(frac_diff, RFI_type='streak', sig=2):
    # Essentially Prewitt filter different RFI shapes

    H = np.zeros([5, 5])

    if RFI_type is 'streak':
        for m in range(5):
            for n in range(5):
                H[m, n] = 1 / (2 * pi * sig**2) * np.exp(-((m - 6)**2 + (n - 6)**2) / (2 * sig**2))

        A = np.zeros([3, frac_diff.shape[2] - 4])
        A[0, :] = 1
        A[2, :] = -1

    smooth = np.zeros(np.array(frac_diff.shape) - [4, 0, 4, 0])
    for m in range(smooth.shape[1]):
        for n in range(smooth.shape[3]):
            for p in range(smooth.shape[0]):
                for q in range(smooth.shape[2]):
                    smooth[p, m, q, n] = np.sum(H * frac_diff[p:p + 5, m, q:q + 5, n])

    edge = np.zeros([smooth.shape[0] - 2, smooth.shape[1], smooth.shape[3]])
    for m in range(edge.shape[1]):
        for n in range(edge.shape[2]):
            for p in range(edge.shape[0]):
                edge[p, m, n] = np.sum(A * smooth[p:p + 3, m, :, n])

    return(smooth, edge)


def match_filter(INS, MS, Nbls, freq_array, sig_thresh, obs):
    # Can pass filter_type a list to check multiple shapes

    def TV_slicer(TV_freqs, freq_array, spw, slices):
        for ch, freq_range in enumerate(TV_freqs):
            if (min(freq_array[spw, :]) < min(freq_range)) or (max(freq_array[spw, :]) > max(freq_range)):
                slices['TV%i' % (ch + 6)] = slice(np.argmin(np.abs(freq_array[spw, :] - min(freq_range))),
                                                  np.argmin(np.abs(freq_array[spw, :] - max(freq_range))))
        return(slices)

    def match_test(MS, Nbls, spw, slc, pol, sig_thresh):
        sliced_arr = MS[:, spw, slc, pol].mean(axis=1)
        N = np.count_nonzero(~MS[:, spw, slc, pol].mask, axis=1)
        thresh = sig_thresh * np.sqrt(np.sum(sigma_calc(Nbls)[:, spw, slc, pol]**2, axis=1)) / N
        if np.any(sliced_arr > thresh):
            t = (sliced_arr / thresh).argmax()
            R = (sliced_arr / thresh).max()
        else:
            t = np.nan
            R = np.nan
        return(t, R)

    TV6_freqs = [1.775e8 - 3.5e6, 1.775e8 + 3.5e6]
    TV7_freqs = [1.845e8 - 3.5e6, 1.845e8 + 3.5e6]
    TV8_freqs = [1.915e8 - 3.5e6, 1.915e8 + 3.5e6]
    TV_freqs = [TV6_freqs, TV7_freqs, TV8_freqs]
    keys = ['streak', 'TV6', 'TV7', 'TV8', 'point']
    event_R = {key: np.zeros([MS.shape[1], MS.shape[3]]) for key in keys}
    R_arr = np.stack([event_R[key] for key in keys])
    while not np.all(np.isnan(R_arr)):
        max_R = -np.inf
        t_max = -1
        key_max = ''
        for m in range(MS.shape[1]):
            slices = {'streak': slice(None), 'TV6': None, 'TV7': None, 'TV8': None}
            slices = TV_slicer(TV_freqs, freq_array, m, slices)
            for n in range(MS.shape[3]):
                for key in keys[:4]:
                    if slices[key]:
                        t, event_R[key][m, n] = match_test(MS, Nbls, m, slices[key], n, sig_thresh)
                        if event_R[key][m, n] > max_R:
                            max_R = event_R[key][m, n]
                            t_max = t
                            key_max = key
                t, f = np.unravel_index(MS[:, m, :, n].argmax(),
                                        MS[:, m, :, n].shape)
                thresh = sig_thresh * sigma_calc(Nbls[t, m, f, n])
                if MS[t, m, f, n] > thresh:
                    event_R['point'][m, n] = MS[t, m, f, n] / thresh
                    if event_R['point'][m, n] > max_R:
                        key_max = None
                else:
                    event_R['point'][m, n] = np.nan
                if key_max:
                    INS[t_max, m, slices[key_max], n] = np.ma.masked
                elif key_max is None:
                    INS[t, m, f, n] = np.ma.masked
        R_arr = np.stack([event_R[key] for key in keys])
        MS = INS / INS.mean(axis=0) - 1

    n, bins = np.histogram(MS[~MS.mask], bins='auto')
    fit = INS_hist_fit(bins, MS, Nbls, sig_thresh)

    return(INS, MS, n, bins, fit)


def narrowband_filter(INS, ch_ignore=None):

    FD_CB = np.zeros(INS.shape)
    for m in range(24):
        INS_CBM = np.array([np.median(INS[:, :, 16 * m:16 * (m + 1), :], axis=2) for k in range(16)]).transpose((1, 2, 0, 3))
        FD_CB[:, :, 16 * m:16 * (m + 1), :] = INS[:, :, 16 * m:16 * (m + 1), :] / INS_CBM - 1
    if ch_ignore is not None:
        INS[:, :, ch_ignore, :] = np.ma.masked
    INS = np.ma.masked_where(np.logical_and(FD_CB > 0.1, ~INS.mask), INS)
    if ch_ignore is not None:
        INS.mask[:, :, ch_ignore, :] = False

    return(INS)


def event_identify(mask, dt=1):
    """
    Search for events which are contiguous in time, belonging to the same spw
    and pol

    dt determines the maximal separation time of flags in order for those flags
    to belong to the same event
    """
    ind = np.where(mask)
    ind = (ind[1], ind[3], ind[0], ind[2])
    ind_stack = np.unique(np.vstack(ind), axis=1)
    diff = ind_stack[:-1].diff(axis=1)
    col_bounds = np.where(np.logical_or(np.logical_or(diff[0, :], diff[1, :]),
                                        diff[2, :] > dt))[0]
    return(ind, event_bound)
