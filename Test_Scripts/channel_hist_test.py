import rfiutil
import plot_lib as pl
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import scipy.stats

indir = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/Golden_Set_8s_Autos/Vis_Avg/Averages'
arrs = glob.glob('%s/*.npy' % (indir))
arrs.sort()
outpath = '/Users/mike_e_dubs/MWA/Test_Plots/channel_hist/'
if not os.path.exists(outpath):
    os.makedirs(outpath)

for path in arrs:
    INS = np.load(path)
    obs = path[len(indir) + 1:len(indir) + 11]
    if not os.path.exists('%s%s/' % (outpath, obs)):
        os.makedirs('%s%s/' % (outpath, obs))

    hist_arr, ks_arr, mu, var = rfiutil.channel_hist(INS)

    for i in range(hist_arr.shape[0]):
        for k in range(hist_arr.shape[1]):
            for m in range(hist_arr.shape[2]):

                fig, ax = plt.subplots(figsize=(14, 8))
                w = np.diff(hist_arr[i, k, m][1])
                x = hist_arr[i, k, m][1][:-1] + 0.5 * w
                gauss = w * INS.shape[0] * scipy.stats.norm.pdf(x, loc=mu[i, k, m],
                                                                scale=np.sqrt(var[i, k, m]))
                pl.one_d_hist_plot(fig, ax, hist_arr[i, k, m][1],
                                   [hist_arr[i, k, m][0], gauss],
                                   labels=['INS', 'Gaussian Fit'], xlog=False,
                                   title='%s f%i pol%i ks = %.4f, p = %.4f' % (obs, k, m, ks_arr[i, k, m][0], ks_arr[i, k, m][1]))
                fig.savefig('%s/%s/INS_channel_hist_spw%i_f%i_p%i.png' % (outpath, obs, i, k, m))
                plt.close(fig)
