from __future__ import division

from SSINS import INS, plot_lib, util, MF
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np

obs = 'zen.2458098.37904.xx.HH'
indir = '/Users/mike_e_dubs/HERA/INS/IDR2_Prelim_Nocut/HERA_IDR2_Prelim_Set_nocut'
outpath = '/Users/mike_e_dubs/General/%s' % obs
read_paths = util.read_paths_construct(indir, None, obs, 'INS')

indir2 = '/Users/mike_e_dubs/HERA/INS/IDR2_OR/HERA_IDR2_Prelim_Set_OR_original'
read_paths_orig = util.read_paths_construct(indir2, 'original', obs, 'INS')
ins2 = INS(read_paths=read_paths_orig, obs=obs, flag_choice='original', outpath=outpath)

ins = INS(read_paths=read_paths, obs=obs, outpath=outpath)
aspect = ins.data.shape[2] / ins.data.shape[0]

fig, ax = plt.subplots(figsize=(16, 9), ncols=2)

#plot_lib.image_plot(fig, ax[0], ins.data[:, 0, :, 0], freq_array=ins.freq_array[0],
                    #cbar_label='Amplitude (UNCALIB)', aspect=aspect, vmax=0.03,
                    #ylabel='Time (10 s)')
plot_lib.image_plot(fig, ax[0], ins.data_ms[:, 0, :, 0], freq_array=ins.freq_array[0],
                    cmap=cm.coolwarm, aspect=aspect, cbar_label='Deviation ($\hat{\sigma}$)',
                    vmin=-5, vmax=5, ylabel='Time (10 s)')


fig_mf, ax_mf = plt.subplots(figsize=(16, 9), ncols=2)
ins.data.mask[:, 0, :82, 0] = True
ins.data.mask[:, 0, -21:, 0] = True
ins.data_ms = ins.mean_subtract(order=1)
mf = MF(ins, sig_thresh=5, N_thresh=0, shape_dict={'TV4': [1.74e8, 1.82e8],
                                                   'TV5': [1.82e8, 1.9e8],
                                                   'TV6': [1.9e8, 1.98e8]})
mf.apply_match_test(apply_N_thresh=False, order=1)

plot_lib.image_plot(fig_mf, ax_mf[0], ins.data[:, 0, :, 0], freq_array=ins.freq_array[0],
                    cbar_label='Amplitude (UNCALIB)', aspect=aspect, vmin=0, vmax=0.03, cmap=cm.viridis,
                    mask_color='white', ylabel='Time (10 s)')
plot_lib.image_plot(fig_mf, ax[1], ins.data_ms[:, 0, :, 0], freq_array=ins.freq_array[0],
                    cbar_label='Deviation ($\hat{\sigma}$)', aspect=aspect, vmin=-5, vmax=5, cmap=cm.coolwarm,
                    mask_color='black', ylabel='Time (10 s)')
fig_mf.savefig('%s/%s_INS_MF.png' % (outpath, obs))
fig.savefig('%s/%s_INS.png' % (outpath, obs))


ins.counts, ins.bins = np.histogram(ins.data_ms[np.logical_not(ins.data.mask)], bins='auto')
ins2.counts, ins2.bins = np.histogram(ins2.data_ms[np.logical_not(ins2.data.mask)], bins='auto')
exp, var = util.hist_fit(ins2.counts, ins2.bins)
N1 = np.sum(ins.counts)
N2 = np.sum(ins2.counts)
Nexp = np.sum(exp)

pdf1 = ins.counts / (N1 * np.diff(ins.bins))
pdf2 = ins2.counts / (N2 * np.diff(ins2.bins))
pdf_exp = exp / (Nexp * np.diff(ins2.bins))

print(np.sum(pdf1 * np.diff(ins.bins)))
print(np.sum(pdf2 * np.diff(ins2.bins)))
print(np.sum(pdf_exp * np.diff(ins2.bins)))

pdf1 = np.append(0, pdf1)
pdf2 = np.append(0, pdf2)
pdf_exp = np.append(0, pdf_exp)
var = np.append(0, var)
err = np.sqrt(var)
err /= (Nexp * np.diff(ins2.bins)[0])
# err = np.sqrt(var) / (np.sum(exp) * np.diff(ins2.bins)[0])
# print(err.shape)

fig3, ax3 = plt.subplots(figsize=(8, 9))
#yerr=np.sqrt(var)[:900]
plot_lib.error_plot(fig3, ax3, ins2.bins[:600], pdf2[:600], drawstyle='steps-post',
                    xlabel='Deviation ($\hat{\sigma}$)', label='XRFI Flagging',
                    legend=True, yscale='log')
plot_lib.error_plot(fig3, ax3, ins2.bins[:600], pdf_exp[:600], drawstyle='steps-post',
                    xlabel='Deviation ($\hat{\sigma}$)', label='Normal Fit',
                    legend=True, yscale='log', yerr=err[:600])
plot_lib.error_plot(fig3, ax3, ins.bins, pdf1, drawstyle='steps-post',
                    xlabel='Deviation ($\hat{\sigma}$)', label='SSINS Flagging',
                    legend=True, yscale='log', ylim=[1e-4, 1],
                    leg_size='xx-large', ylabel='PDF')
fig3.savefig('%s/%s_INS_Hist_Compare.png' % (outpath, obs))
