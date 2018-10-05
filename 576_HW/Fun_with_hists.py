from SSINS import SS, INS
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

ss = SS(obs='1061312272', outpath='/Users/mike_e_dubs/576',
        inpath='/Users/mike_e_dubs/MWA/Data/uvfits/1061313128.uvfits',
        bad_time_indices=[0, -1, -2, -3], read_kwargs={'ant_str': 'cross'})
ss.VDH_prepare(bins='auto', fit_hist=True)
N = ss.UV.data_array.size
bins = ss.VDH.bins[0]
centers = bins[:-1] + 0.5 * np.diff(bins)
mu = np.mean(ss.UV.data_array)
var = np.var(ss.UV.data_array)
r_scale = np.sqrt(0.5 * np.mean(ss.UV.data_array**2))
norm = scipy.stats.norm
rayleigh = scipy.stats.rayleigh
gauss_fit = N * (norm.cdf(bins[1:], loc=mu, scale=np.sqrt(var)) - norm.cdf(bins[:-1], loc=mu, scale=np.sqrt(var)))
rayleigh_fit = N * (rayleigh.cdf(bins[1:], scale=r_scale) - rayleigh.cdf(bins[:-1], scale=r_scale))

fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(centers, ss.VDH.counts[0], drawstyle='steps-mid', label='Background')
ax.plot(centers, ss.VDH.fits[0], drawstyle='steps-mid', label='Rayleigh Mixture Fit')
ax.plot(centers, gauss_fit, drawstyle='steps-mid', label='Gaussian Fit')
ax.plot(centers, rayleigh_fit, drawstyle='steps-mid', label='Single Rayleigh Fit')
ax.legend()
ax.set_xlabel('Amplitude (UNCALIB)')
ax.set_ylabel('Counts')
ax.set_yscale('log', nonposy='clip')
ax.set_ylim([0.1, 10 * np.amax(ss.VDH.counts[0])])
fig.savefig('%s/Rayleigh_Dist_RFI.png' % ss.outpath)
