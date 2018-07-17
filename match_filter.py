import numpy as np
from scipy.special import erfcinv
import argparse
import glob
import scipy.stats
import os


def shape_slicer(shape_dict, freq_array, spw):
    slice_dict = {}
    sig_thresh = {}
    for shape in shape_dict:
        # Ask if any of the shape is in the freq_array
        if (min(freq_array[spw, :]) < min(shape_dict[shape])) or (max(freq_array[spw, :]) > max(shape_dict[shape])):
            # Rewrite the shape_dict entry in terms of channel numbers which match the shape best
            slice_dict[shape] = slice(np.argmin(np.abs(freq_array[spw, :] - min(shape_dict[shape]))),
                                      np.argmin(np.abs(freq_array[spw, :] - max(shape_dict[shape]))))
            N = slice_dict[shape].indices(INS.shape[2])[1] - slice_dict[shape].indices(INS.shape[2])[0]
            sig_thresh[shape] = np.sqrt(2) * erfcinv(float(N) / np.prod(INS.shape))
        # Set the slice to None if the shape is not at all in the freq_array
        else:
            slice_dict[shape] = None
    slice_dict['streak'] = slice(None)
    slice_dict['point'] = slice(None)
    sig_thresh['streak'] = np.sqrt(2) * erfcinv(float(INS.shape[2]) / np.prod(INS.shape))
    sig_thresh['point'] = np.sqrt(2) * erfcinv(1. / np.prod(INS.shape))
    return(slice_dict, sig_thresh)


def match_test(MS, Nbls, spw, sig_thresh, slice_dict):
    # Treat point and slices separately
    R_max = -np.inf
    t_max = None
    f_max = None
    for shape in slice_dict:
        if slice_dict[shape] is not None:
            if shape is 'point':
                t, f, p = np.unravel_index(np.absolute(MS[:, spw] / sig_thresh[shape]).argmax(), MS[:, spw].shape)
                R = np.absolute(MS[t, spw, f, p] / sig_thresh[shape])
                f = slice(f, f + 1)
            else:
                # Average across the shaoe in question specified by slc - output is 1D
                N = np.count_nonzero(np.logical_not(MS[:, spw, slice_dict[shape]].mask), axis=1)
                sliced_arr = np.absolute(MS[:, spw, slice_dict[shape]].mean(axis=1)) * np.sqrt(N)
                # Gauss dist, so expected width is as below
                t, p = np.unravel_index((sliced_arr / sig_thresh[shape]).argmax(), sliced_arr.shape)
                f = slice_dict[shape]
                R = sliced_arr[t, p] / sig_thresh[shape]
            if R > 1:
                if R > R_max:
                    t_max, f_max, R_max = (t, f, R)

    return(t_max, f_max, R_max)


def hist_prob(bins, dist='norm', args=()):

    cdf = getattr(scipy.stats, dist).cdf
    prob = cdf(bins[1:], *args) - cdf(bins[:-1], *args)
    return(prob)


def hist_construct(MS, event, sig_thresh):
    data = MS[:, event[0], event[1]]
    N = np.count_nonzero(np.logical_not(data.mask), axis=1)
    data = data.mean(axis=1) * np.sqrt(N)
    bins = np.linspace(-sig_thresh, sig_thresh, num=2 * int(np.ceil(2 * sig_thresh + 1)))
    if np.amin(data) < -sig_thresh:
        bins = np.insert(bins, 0, np.amin(data))
    if np.amax(data) > sig_thresh:
        bins = np.append(bins, np.amax(data))
    n, _ = np.histogram(data[np.logical_not(data.mask)], bins=bins)
    return(n, bins)


def bin_combine(counts, bins, weight='exp', thresh=5):

    if weight is 'exp':
        c_cond = np.logical_or(exp < thresh, counts < thresh)
    elif weight is 'var':
        var = np.sum(counts) * prob * (1 - prob)
        c_cond = var < thresh

    while np.any(c_cond):
        ind = np.where(c_cond)[0][0]
        # If the index is zero, add the bin on the right and delete the bin on
        # the right. Else, add the bin on the left and delete the bin on the left.
        for dat in [counts, exp, prob]:
            dat[ind] += dat[ind + (-1)**(bool(ind))]
            dat = np.delete(dat, ind + (-1)**(bool(ind)))
        bins = np.delete(bins, ind + (-1)**(bool(ind)))
        if weight is 'exp':
            c_cond = np.logical_or(exp < 5, counts < 5)
        elif weight is 'var':
            var = np.sum(counts) * prob * (1 - prob)
            c_cond = var < 1

    return(counts, exp, prob, bins)


def chisq_test(INS, MS, sig_thresh, event, weight='exp'):

    counts, bins = hist_construct(MS, event, sig_thresh)
    N = np.sum(counts)
    prob = hist_prob(bins)
    exp = N * prob
    counts, exp, prob, bins = bin_combine(counts, exp, prob, bins, weight=weight)
    var = N * prob * (1 - prob)
    if weight is 'exp':
        stat, p = scipy.stats.chisquare(counts, exp, ddof=2)
    elif weight is 'var':
        stat = np.sum((counts - exp)**2 / var)
        p = scipy.stats.chi2.isf(stat, len(var) - 3)
    alpha = erfc(sig_thresh / np.sqrt(2))
    if p < alpha:
        INS[:, event[0], event[1]] = np.ma.masked

    return(INS)

def match_filter(INS, MS, Nbls, outpath, freq_array, obs, sig_thresh, shape_dict={},
                 chisq=False, N=0):

    events = []
    hists = []
    count = 1
    total = 0
    base = '%s/match_filter/%s' % (outpath, obs)
    C = 4 / np.pi - 1
    while count:
        count = 0
        for m in range(MS.shape[1]):
            slice_dict, sig_thresh = shape_slicer(shape_dict, freq_array, m)
            t_max, f_max, R_max = match_test(MS, Nbls, m, sig_thresh, slice_dict)
            if R_max > -np.inf:
                count += 1
                total += 1
                INS[t_max, m, f_max] = np.ma.masked
                events.append((m, f_max, t_max))
        MS = (INS / INS.mean(axis=0) - 1) * np.sqrt(Nbls / C)
        np.ma.dump(INS, '%s_INS_mf_%i.npym' % (base, total))
        np.ma.dump(MS, '%s_MS_mf_%i.npym' % (base, total))
        if count:
            for p in range(1, count + 1):
                hists.append(hist_construct(MS, events[-p], sig_thresh['point']))

    if events:
        un_events = list(set([event[:-1] for event in events]))
        for event in un_events:
            if chisq:
                INS = chisq_test(INS, MS, sig_thresh, event, weight='var')
                MS = (INS / INS.mean(axis=0) - 1) * np.sqrt(Nbls / C)
                np.ma.dump(INS, '%s_INS_mf_chisq.npym' % (base))
                np.ma.dump(MS, '%s_MS_mf_chisq.npym' % (base))
            if N:
                if np.count_nonzero(np.logical_not(INS.mask[:, event[0], event[1], 0]), axis=0) < N:
                    INS[:, event[0], event[1]] = np.ma.masked
                MS = (INS / INS.mean(axis=0) - 1) * np.sqrt(Nbls / C)
                np.ma.dump(INS, '%s_INS_mf_Nsamp.npym' % (base))
                np.ma.dump(MS, '%s_MS_mf_Nsamp.npym' % (base))
        hists = [x for _, x in sorted(zip(events, hists), key=lambda pair: pair[0][:-1])]
        events = sorted(events, key=lambda events: events[:-1])
        events = np.vstack(events)

    np.ma.dump(INS, '%s_INS_mf_final.npym' % (base))
    np.ma.dump(MS, '%s_MS_mf_final.npym' % (base))
    np.save('%s_events.npy' % (base), events)
    np.save('%s_hists.npy' % (base), hists)

    return(INS, MS, events, hists)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inpath", nargs=1, action='store', help="The location of the original spectra to be flagged")
    parser.add_argument("outpath", nargs=1, action='store', help="Destination for the results")
    parser.add_argument("freq_array", nargs=1, action='store', help="Path to freq_arr.npy")
    parser.add_argument('sig_thresh', nargs=1, action='store', type=float, help="The z-score you would like to threshold")
    parser.add_argument('--keys', nargs='*', help="The names of the shapes", default=[])
    parser.add_argument('--lows', nargs='*', action='store', type=float, default=[],
                        help="The lower bound of the shapes referred to by the keys")
    parser.add_argument('--highs', nargs='*', action='store', type=float, default=[],
                        help="The upper bounds of the shapes referred to by the keys")
    parser.add_argument('-c', action='store_true', help="Switch to do the chisquare test", default=False)
    parser.add_argument('-N', nargs=1, action='store', type=int, default=0,
                        help="Switch to flag remainder of times if Nremain < N")
    args = parser.parse_args()

    freq_arr = np.load(args.freq_array[0])
    INS_arrs = glob.glob('%s/*INS*' % (args.inpath[0]))
    INS_arrs.sort()
    Nbls_arrs = glob.glob('%s/*Nbls*' % (args.inpath[0]))
    Nbls_arrs.sort()

    shape_dict = dict(zip(args.keys, zip(args.lows, args.highs)))
    mf_kwargs = {'shape_dict': shape_dict,
                 'chisq': args.c,
                 'N': args.N[0]}

    C = 4 / np.pi - 1
    if not os.path.exists(args.outpath[0]):
        os.makedirs(args.outpath[0])

    for INS_path, Nbls_path in zip(INS_arrs, Nbls_arrs):
        INS = np.ma.masked_array(np.load(INS_path))
        Nbls = np.load(Nbls_path)
        MS = (INS / INS.mean(axis=0) - 1) * np.sqrt(Nbls / C)
        obs = INS_path[len(args.inpath[0]):INS_path.find('_INS.npym')]
        mf_args = (INS, MS, Nbls, args.outpath[0], freq_arr, obs, args.sig_thresh[0])

        INS, MS, events, hists = match_filter(*mf_args, **mf_kwargs)
