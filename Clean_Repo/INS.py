import numpy as np
from scipy.special import erfcinv


class INS:
    """
    This incoherent noise spectrum class, formed from an RFI object as an input.
    Outputs a data array
    """

    def __init__(self, RFI=None, data=None, Nbls=None, freq_array=None,
                 pols=None, vis_units=None, obs=None, outpath=None):

        args = (data, Nbls, freq_array, pols, vis_units, obs, outpath)
        assert RFI is not None or all([arg is not None for arg in args]), \
            'Insufficient input given. Supply an instance of the RFI class or read in appropriate data.'

        if RFI is not None:
            self.data = RFI.UV.data_array.mean(axis=1)
            self.Nbls = np.count_nonzero(np.logical_not(RFI.UV.data_array.mask),
                                         axis=1)
            self.freq_array = RFI.UV.freq_array
            self.pols = RFI.pols
            self.vis_units = RFI.UV.vis_units
            self.obs = RFI.obs
            self.outpath = RFI.outpath
        elif not any([arg is None for arg in args]):
            self.data = data
            self.Nbls = Nbls
            self.freq_array = freq_array
            self.pols = pols
            self.vis_units = vis_units
            self.obs = obs
            self.outpath = outpath
        self.data_ms = self.mean_subtract()
        self.counts, self.bins, self.sig_thresh = self.hist_make()
        self.events = []
        self.event_hists = []
        self.chisq_events = []
        self.chisq_hists = []

    def mean_subtract(self):

        C = 4 / np.pi - 1
        MS = (self.data / self.data.mean(axis=0) - 1) * np.sqrt(self.Nbls / C)
        return(MS)

    def hist_make(self, sig_thresh=None, event=None):

        if sig_thresh is None:
            sig_thresh = np.sqrt(2) * erfcinv(1. / np.prod(self.data.shape))
        bins = np.linspace(-self.sig_thresh, self.sig_thresh,
                           num=int(2 * np.ceil(2 * self.sig_thresh)))
        if event is None:
            dat = self.data_ms
        else:
            N = np.count_nonzero(np.logical_not(self.data_ms.mask[:, event[0], event[1]]), axis=1)
            dat = self.data_ms[:, event[0], event[1]].mean(axis=1) * np.sqrt(N)
        if dat.min() < -sig_thresh:
            bins = np.insert(bins, 0, dat.min())
        if dat.max() > sig_thresh:
            bins = np.append(bins, dat.max())
        counts, _ = np.histogram(dat[np.logical_not(dat.mask)],
                                 bins=bins)

        return(counts, bins, sig_thresh)
