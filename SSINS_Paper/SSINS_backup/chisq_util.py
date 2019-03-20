def hist_fit(counts, bins, dist='norm'):
    """
    Given a histogram, draws the expected counts and variance for the specified
    distribution. Must be a scipy distribution.
    """
    N = np.sum(counts)
    p = getattr(scipy.stats, dist).cdf(bins[1:]) - getattr(scipy.stats, dist).cdf(bins[:-1])
    exp = N * p
    var = N * p * (1 - p)

    return(exp, var)


def bin_combine(counts, bins, weight='var', thresh=1, dist='norm'):

    """
    Combines bins from the outside in until all bins have a weight which exceeds
    thresh. Used for making a reasonable chisquare calculation.

    Arguments: counts: The counts in the histogram bins

               bins: The bin edges for the histogram

               weight: Choices are 'var' or 'exp'

                       'var': The expected variance in each bin must exceed
                              thresh

                       'exp': The expected counts AND the counts must exceed
                              thresh in each bin

               dist: The scipy distribution to calculate expected counts/bins
                      with
    """

    exp, var = hist_fit(counts, bins, dist=dist)
    c_com, b_com = (np.copy(counts), np.copy(bins))

    if weight is 'exp':
        # Sum the counts and make sure a valid bin is possible at all
        S = np.sum(counts)
        c_cond = np.logical_or(exp < thresh, counts < thresh)
    elif weight is 'var':
        # Sum the var and make sure a valid bin is possible at all
        S = np.sum(var)
        c_cond = var < thresh
    if S > thresh:
        while np.any(c_cond) and len(c_com) > 4:
            c_com[1] += c_com[0]
            c_com[-2] += c_com[-1]
            c_com = c_com[1:-1]
            b_com = np.delete(b_com, (1, len(b_com) - 2))
            exp, var = hist_fit(c_com, b_com, dist=dist)
            if weight is 'exp':
                c_cond = np.logical_or(exp < thresh, c_com < thresh)
            elif weight is 'var':
                c_cond = var < thresh

    return(c_com, b_com)


def chisq(counts, bins, weight='var', thresh=1, dist='norm'):

    """
    Calculates a chisq statistic given a distribution and a chosen weight
    scheme.
    """

    counts, bins = bin_combine(counts, bins, weight=weight, thresh=thresh, dist=dist)
    exp, var = hist_fit(counts, bins, dist=dist)
    if weight is 'exp':
        S = np.sum(counts)
        stat, p = scipy.stats.chisquare(counts, exp, ddof=2)
    elif weight is 'var':
        S = np.sum(var)
        stat = np.sum((counts - exp)**2 / var)
        p = scipy.stats.chi2.sf(stat, len(var) - 3)
    if S < thresh:
        stat, p = np.nan, np.nan

    return(stat, p)
