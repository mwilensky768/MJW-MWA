import numpy as np
from math import floor, ceil, log10, pi, log, sqrt
from scipy.special import erfinv
from scipy.integrate import simps


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


def match_filter(INS, MS, Nbls, freq_array, sig_thresh, shape_dict):
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
                    thresh = sig_thresh * sigma_calc(Nbls[:, spw, :, pol])
                    t, f = np.unravel_index((MS[:, spw, :, pol] / thresh).argmax(), MS[:, spw, :, pol].shape)
                    R = MS[t, spw, f, pol] / thresh[t, f]
                else:
                    # Average across the shaoe in question specified by slc - output is 1D
                    sliced_arr = MS[:, spw, slice_dict[shape], pol].mean(axis=1)
                    N = np.count_nonzero(~MS[:, spw, slice_dict[shape], pol].mask, axis=1)
                    # Gauss dist, so expected width is as below
                    thresh = sig_thresh * \
                        np.sqrt(np.sum(sigma_calc(Nbls[:, spw, slice_dict[shape], pol])**2, axis=1)) / N
                    t, f = ((sliced_arr / thresh).argmax(), slice_dict[shape])
                    R = sliced_arr[t] / thresh[t]
                if R > 1:
                    if R > R_max:
                        t_max, f_max, R_max = (t, f, R)

        return(t_max, f_max, R_max)

    R_max = 0
    event_count = []
    while R_max > -np.inf:
        for m in range(MS.shape[1]):
            slice_dict = shape_slicer(shape_dict, freq_array, m)
            for n in range(MS.shape[3]):
                t_max, f_max, R_max = match_test(MS, Nbls, m, n, sig_thresh, slice_dict)
                if R_max > -np.inf:
                    INS[t_max, m, f_max, n] = np.ma.masked
                    event_count.append((t_max, m, f_max, n))
        if R_max > -np.inf:
            MS = INS / INS.mean(axis=0) - 1

    n, bins = np.histogram(MS[~MS.mask], bins='auto')
    fit = INS_hist_fit(bins, MS, Nbls, sig_thresh)

    return(INS, MS, n, bins, fit, event_count)


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


def rayleigh_convolve(N, MAX):
    # Convolve a the median-subtracted rayleigh distribution N times by using
    # the convolution theorem
    M = 1001
    x = np.linspace(-1, MAX, num=M)
    Fs = 1. / x[1] - x[0]
    f = np.linspace(0, (1 - 1. / M) * Fs, num=M)
    pdf = 2 * np.log(2) * (x + 1) * np.exp(-np.log(2) * (x + 1)**2)
    cf = np.zeros(M)
    # Calculate the fourier transform using Simpson's rule
    for m, f_0 in enumerate(f):
        cf[m] = (simps(np.exp(1j * x * f_0) * pdf, x=x))**N
    for m, x_0 in enumerate(x):
        pdf[m] = simps(np.exp(-1j * x_0 * f) * cf, x=f) / (2 * pi)

    return(x, pdf)


def event_identify(mask, shape_dict, dt=1):
    """
    Search for events which are contiguous in time, belonging to the same spw
    and pol

    dt determines the maximal separation time of flags in order for those flags
    to belong to the same event
    """
    # Find the indices of the masked data in a tuple
    ind = np.where(mask)
    # Reorder the tuple to spw, pol, time, freq
    ind = (ind[1], ind[3], ind[0], ind[2])
    # stack them vertically and use unique to sort
    ind_stack = np.unique(np.vstack(ind), axis=1)
    # take the difference in the stack
    diff = ind_stack[:-1].diff(axis=1)
    # Find the column index for the last member of an event
    col_bounds = np.where(np.logical_or(np.logical_or(diff[0, :], diff[1, :]),
                                        diff[2, :] > dt))[0]

    return(ind, col_bounds)


def emp_pdf_flag(size, data, bins, thresh=10, scale=1. / np.sqrt(np.log(2))):
    A = np.random.rayleigh(scale=scale, size=size).mean(axis=0)
    model, _ = np.histogram(A, bins=bins, density=True) * len(data.flatten())
    max_loc = bins[:-1][model.argmax()]
    cutoff = min(bins[(model < 1) & (bins[:-1] > max_loc)])
    data[data > cutoff] = np.ma.masked

    return(cutoff, data)
