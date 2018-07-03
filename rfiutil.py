import numpy as np
from math import floor, ceil, log10, pi, log, sqrt
from scipy.special import erfinv
from scipy.integrate import simps
import scipy.stats
import time


def save(object, path, mask=False):

    if mask:
        np.ma.dump(object, '%s.npym' % (path))
    else:
        np.save('%s.npy' % (path), object)


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


def match_filter(INS, MS, Nbls, outpath, freq_array, sig_thresh=4, shape_dict={}, dt=1):
    """
    shape_dict is a dictionary whose key is the name of the shape as a string
    and each entry is a tuple of frequencies given in the units of the freq_array.
    """

    def shape_slicer(shape_dict, freq_array, spw):
        slice_dict = {}
        for shape in shape_dict:
            # Ask if any of the shape is in the freq_array
            if (min(freq_array[spw, :]) < min(shape_dict[shape])) or (max(freq_array[spw, :]) > max(shape_dict[shape])):
                # Rewrite the shape_dict entry in terms of channel numbers which match the shape best
                slice_dict[shape] = slice(np.argmin(np.abs(freq_array[spw, :] - min(shape_dict[shape]))),
                                          np.argmin(np.abs(freq_array[spw, :] - max(shape_dict[shape]))))
            # Set the slice to None if the shape is not at all in the freq_array
            else:
                slice_dict[shape] = None
        slice_dict['streak'] = slice(None)
        slice_dict['point'] = slice(None)
        return(slice_dict)

    def match_test(MS, Nbls, spw, pol, sig_thresh, slice_dict):
        # Treat point and slices separately
        R_max = -np.inf
        t_max = None
        f_max = None
        for shape in slice_dict:
            if slice_dict[shape] is not None:
                if shape is 'point':
                    t, f = np.unravel_index(np.absolute(MS[:, spw, :, pol] / sig_thresh).argmax(), MS[:, spw, :, pol].shape)
                    R = np.absolute(MS[t, spw, f, pol] / sig_thresh)
                    f = slice(f, f + 1)
                else:
                    # Average across the shaoe in question specified by slc - output is 1D
                    N = np.count_nonzero(np.logical_not(MS[:, spw, slice_dict[shape], pol].mask), axis=1)
                    sliced_arr = np.absolute(MS[:, spw, slice_dict[shape], pol].mean(axis=1)) * np.sqrt(N)
                    # Gauss dist, so expected width is as below
                    t, f = ((sliced_arr / sig_thresh).argmax(), slice_dict[shape])
                    R = sliced_arr[t] / sig_thresh
                if R > 1:
                    if R > R_max:
                        t_max, f_max, R_max = (t, f, R)

        return(t_max, f_max, R_max)

    def hist_construct(MS, event):
        data = MS[:, event[0], event[2], event[1]]
        N = np.count_nonzero(np.logical_not(data.mask), axis=1)
        data = data.mean(axis=1) * np.sqrt(N)
        n, bins = np.histogram(data[np.logical_not(data.mask)], bins=np.linspace(-4, 4, num=9))
        return(n, bins)

    events = []
    hists = []
    count = 1
    while count > 0:
        count = 0
        for m in range(MS.shape[1]):
            slice_dict = shape_slicer(shape_dict, freq_array, m)
            for n in range(MS.shape[3]):
                t_max, f_max, R_max = match_test(MS, Nbls, m, n, sig_thresh, slice_dict)
                if R_max > -np.inf:
                    count += 1
                    INS[t_max, m, f_max, n] = np.ma.masked
                    events.append((m, n, f_max, t_max))
        MS = (INS / INS.mean(axis=0) - 1) * np.sqrt(Nbls / (4 / np.pi - 1))
        if count:
            for p in range(1, count + 1):
                hists.append(hist_construct(MS, events[-p]))

    if events:
        hists = [x for _, x in sorted(zip(events, hists), key=lambda pair: pair[0][:-1])]
        events = sorted(events, key=lambda events: events[:-1])
        events = np.vstack(events)

    obj_tup = (INS, MS, events)
    name_tup = ('INS_mask', 'INS_MS_mask', 'INS_events')
    mask_tup = (True, True, False, False, False, False)

    for obj, name, mask in zip(obj_tup, name_tup, mask_tup):
        save(obj, '%s_%s' % (outpath, name), mask=mask)

    return(INS, MS, events, hists)


def narrowband_filter(INS, ch_ignore=None):

    FD_CB = np.zeros(INS.shape)
    for m in range(24):
        INS_CBM = np.array([np.median(INS[:, :, 16 * m:16 * (m + 1), :], axis=2) for k in range(16)]).transpose((1, 2, 0, 3))
        FD_CB[:, :, 16 * m:16 * (m + 1), :] = INS[:, :, 16 * m:16 * (m + 1), :] / INS_CBM - 1
    if ch_ignore is not None:
        INS[:, :, ch_ignore, :] = np.ma.masked
    INS = np.ma.masked_where(np.logical_and(FD_CB > 0.1, np.logical_not(INS.mask)), INS)
    if ch_ignore is not None:
        INS.mask[:, :, ch_ignore, :] = False

    return(INS)


def ks_test(MS, mode='approx'):

    ks_arr = np.zeros(MS.shape[1:], dtype=object)
    for i in range(MS.shape[1]):
        for k in range(MS.shape[2]):
            for m in range(MS.shape[3]):
                ks_arr[i, k, m] = scipy.stats.kstest(MS[:, i, k, m][np.logical_not(MS[:, i, k, m].mask)],
                                                     'norm', mode=mode)

    return(ks_arr)


def emp_pdf(Nt, Nf, Nbls, bins, scale=1, dist='rayleigh'):

    A = getattr(np.random, dist)(size=(Nt, Nbls, Nf), scale=scale).mean(axis=(0, 2))
    sim, _ = np.histogram(A, bins=bins)

    return(A, sim)


def event_compile(events, dt=1):

    # Sort the flagged events basically in chronological order and stack them
    if events:
        perm = sorted(range(len(events)), key=events.__getitem__())
        events.sort()
        event_stack = np.vstack(events)
        # print(event_stack)
        # Find where the spw/pol/freqs do not agree OR where time separation is greater than dt
        # Add one so that the bounds mark the start of a new event
        row_bounds = np.where(np.logical_or(np.any(event_stack[:-1, :-1] != event_stack[1:, :-1], axis=1),
                                            np.diff(event_stack[:, -1], axis=0) > dt))[0]
        # insert zero and N_event so that making slices is more straightforward than otherwise
        # first slice goes from zero, last slice goes to the end
        row_bounds = np.insert(row_bounds, 0, -1)
        row_bounds = np.append(row_bounds, len(event_stack) - 1)
        # print(row_bounds)

        # Generate a list of time_slices to be hstacked with the events
        time_slices = []
        for m, row in enumerate(row_bounds[:-1]):
            # Has to shift up from the row entry to start, then go to the row_bound, and add 1 to the time obtained
            time_slices.append(slice(event_stack[row + 1, -1], event_stack[row_bounds[m + 1], -1] + 1))
        events = np.hstack((event_stack[row_bounds[:-1] + 1, :-1], np.array(time_slices, ndmin=2).transpose()))
        # print(events)

        return(events, perm)
