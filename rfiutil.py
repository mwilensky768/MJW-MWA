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


def hist_fit(obs, bins, values, flags, writepath='',
             flag_slice='All', bin_window=[0, 1e3], fit_type='rayleigh'):

    bin_widths = np.diff(bins)
    bin_centers = bins[:-1] + 0.5 * bin_widths
    fit = np.zeros(len(bins) - 1)

    if fit_type is 'rayleigh':
        m = np.copy(fit)
        sig_sq_array = np.zeros(values.shape[2:])
        # Calculate maximum likelihood estimator for each frequency and polarization
        for l in range(values.shape[3]):
            for k in range(values.shape[2]):
                n, bins = np.histogram(values[:, :, k, l][flags[:, :, k, l]], bins=bins)
                N = np.sum(n)
                m += n
                # Only use data within the fit window
                data_cond = np.logical_and(np.logical_and(min(bin_window) < values[:, :, k, l],
                                           values[:, :, k, l] < max(bin_window)), flags[:, :, k, l])
                N_fit = np.count_nonzero(data_cond)
                # Do not calculate a sigma if nothing belonging to the flag slice is in the window
                if N_fit > 0:
                    # Calculate the MLE and histogram according to the MLE
                    sig_sq = 0.5 * np.sum(values[:, :, k, l][data_cond]**2) / N_fit
                    fit += N_fit * bin_widths * bin_centers * \
                        np.exp(-bin_centers**2 / (2 * sig_sq)) / sig_sq
                else:
                    sig_sq = np.nan
                sig_sq_array[k, l] = sig_sq

        np.save('%s%s_%s_sigsq.npy' % (writepath, obs, flag_slice),
                sig_sq_array)

    if fit_type is 'normal':
        data_cond = np.logical_and(np.logical_and(min(bin_window) < values,
                                                  values < max(bin_window)),
                                   flags)
        N = len(values[data_cond])
        mu = np.mean(values[data_cond])
        sigma_sq = np.var(values[data_cond])
        fit = N * bin_widths / np.sqrt(2 * pi * sigma_sq) * \
            np.exp(-((bin_centers - mu) ** 2) / (2 * sigma_sq))
        m = np.histogram(values.flatten())

    return(m, fit)


def INS_outlier_flag(obs, INS, frac_diff, Nbls, flag_slice='All', amp_avg='Amp',
                     write=False, writepath='', wind_len=16, st_iter=0,
                     sir_agg=0.2):

    # This is how far noisy data will extend
    bin_max = sqrt((pi - 4) / (pi * Nbls) * 2 *
                   log(sqrt(pi) / (4 * len(INS[~INS.mask]) ** (2.0 / 3) * erfinv(0.5))))

    # Flag the greatest outlier and recalculate the frac_diff in each iteration to determine new outliers
    while np.max(np.absolute(frac_diff)) > bin_max:
        INS[np.unravel_index(np.absolute(frac_diff).argmax(), INS.shape)] = np.ma.masked
        frac_diff = INS / INS.mean(axis=0) - 1

    # Make new histogram afterward, with fit
    n, bins = np.histogram(frac_diff[np.logical_not(frac_diff.mask)], bins='auto')
    _, fit = hist_fit(obs, bins, frac_diff, np.logical_not(frac_diff.mask),
                      bin_window=[-bin_max, bin_max], fit_type='normal')

    base = '%s%s_%s_%s' % (writepath, obs, flag_slice, amp_avg)
    np.ma.dump(INS, '%s_INS.npym' % (base))
    np.ma.dump(frac_diff, '%s_INS_frac_diff.npym' % (base))
    np.save('%s_INS_counts.npy' % (base), n)
    np.save('%s_INS_bins.npy' % (base), bins)
    np.save('%s_INS_fit.npy' % (base), fit)

    return(INS, frac_diff, n, bins, fit)
