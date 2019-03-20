from SSINS import INS, util, plot_lib, MF
from matplotlib import cm
import matplotlib.pyplot as plt
import os
import numpy as np

obs = 'zen.2458101.52817.xx.HH'
indir = '/Users/mike_e_dubs/HERA/INS/IDR2_Prelim_Nocut/HERA_IDR2_Prelim_Set_nocut'
indir2 = '/Users/mike_e_dubs/HERA/INS/IDR2_OR/HERA_IDR2_Prelim_Set_OR_original'
outpath = '/Users/mike_e_dubs/General/%s' % obs
hist_shit = False

read_paths = util.read_paths_construct(indir, None, obs, 'INS')
ins = INS(read_paths=read_paths, obs=obs, outpath=outpath)
fig1, ax1 = plt.subplots(figsize=(14, 8), nrows=3)
fig2, ax2 = plt.subplots(figsize=(14, 8), nrows=2)

im_kwargs = [{'cbar_label': 'Amplitude (UNCALIB)'},
             {'vmin': -5, 'vmax': 5, 'cmap': cm.coolwarm, 'mask_color': 'black',
              'cbar_label': 'Deviation ($\hat{\sigma}$)'}]

data = [ins.data, ins.data_ms]
for i in range(2):
    plot_lib.image_plot(fig1, ax1[i], data[i][:, 0, :, 0],
                        freq_array=ins.freq_array[0], **im_kwargs[i])
if not os.path.exists(outpath):
    os.makedirs(outpath)


shape_dict = {}
ins.data_ms = ins.mean_subtract(order=0)
mf = MF(ins, sig_thresh=5, N_thresh=20, shape_dict=shape_dict)
mf.apply_match_test(apply_N_thresh=True, order=0)
plot_lib.image_plot(fig2, ax2[0], ins.data_ms[:, 0, :, 0],
                    freq_array=ins.freq_array[0], **im_kwargs[1])

read_paths_orig = util.read_paths_construct(indir2, 'original', obs, 'INS')
plot_lib.image_plot(fig1, ax1[2], ins.data_ms[:, 0, :, 0],
                    freq_array=ins.freq_array[0], **im_kwargs[1])
fig1.savefig('%s/%s_INS_Basic.png' % (outpath, obs))

if hist_shit:
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    ins.counts, ins.bins = np.histogram(ins.data_ms[np.logical_not(ins.data.mask)], bins='auto')
    ins2.counts, ins2.bins = np.histogram(ins2.data_ms[np.logical_not(ins2.data.mask)], bins='auto')
    ins.counts /= (np.sum(ins.counts) * np.diff(ins.bins))
    ins.counts /= (np.sum(ins2.counts) * np.diff(ins2.bins))
    ins.counts = np.append(0, ins.counts)
    ins2.counts = np.append(0, ins2.counts)
    exp, var = util.hist_fit(ins2.counts, ins2.bins)
    exp /= (np.sum(exp) * np.diff(ins2.bins))
    var /= (np.sum(exp) * np.diff(ins2.bins))**2
    exp = np.append(0, exp)
    var = np.append(0, var)
    print(np.amax(ins.data_ms))
    print(np.amax(ins2.data_ms))
    plot_lib.error_plot(fig3, ax3, ins.bins, ins.counts, drawstyle='steps-post',
                        xlabel='Deviation ($\hat{\sigma}$)', label='SSINS Flagging',
                        legend=True, yscale='log')
    plot_lib.error_plot(fig3, ax3, ins2.bins, ins2.counts, drawstyle='steps-post',
                        xlabel='Deviation ($\hat{\sigma}$)', label='XRFI Flagging',
                        legend=True, yscale='log')
    plot_lib.error_plot(fig3, ax3, ins2.bins, exp, yerr=np.sqrt(var), drawstyle='steps-post',
                        xlabel='Deviation ($\hat{\sigma}$)', label='Normal Fit',
                        legend=True, yscale='log', ylim=[0.1, 10 * np.amax(ins2.counts)])
    fig3.savefig('%s/%s_INS_Hist_Compare.png' % (outpath, obs))
