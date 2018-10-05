from SSINS import INS, MF, util
from SSINS import plot_lib as pl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import os

rawpath = '/Users/mike_e_dubs/HERA/INS/IDR2_Prelim_Nocut/HERA_IDR2_Prelim_Set_nocut'
ORpath = '/Users/mike_e_dubs/HERA/INS/IDR2_OR/HERA_IDR2_Prelim_Set_OR_original'
obslist_path = '/Users/mike_e_dubs/python_stuff/MJW-MWA/Obs_Lists/HERA_IDR2_Prelim_obs.txt'
outpath = '/Users/mike_e_dubs/HERA/INS/Datacon_0'
figpath = '%s/figs' % outpath
if not os.path.exists(figpath):
    os.makedirs(figpath)

shape_dict = {'TV4': [1.74e8, 1.82e8],
              'TV5': [1.82e8, 1.9e8],
              'TV6': [1.9e8, 1.98e8],
              'dig1': [1.125e8, 1.15625e8],
              'dig2': [1.375e8, 1.40625e8],
              'dig3': [1.625e8, 1.65625e8],
              'dig4': [1.875e8, 1.90625e8]}

obslist = util.make_obslist(obslist_path)

for obs in obslist:
    raw_reads = util.read_paths_construct(rawpath, 'None', obs, 'INS')
    OR_reads = util.read_paths_construct(ORpath, 'original', obs, 'INS')
    raw_ins = INS(obs=obs, flag_choice=None, read_paths=raw_reads, order=0,
                  outpath=outpath)
    OR_ins = INS(obs=obs, flag_choice='original', read_paths=OR_reads)

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(14, 8))
    fig.suptitle('%s Incoherent Noise Spectrum Comparison' % obs)
    attr = ['data', 'data_ms']
    cbar_label = ['UNCALIB', 'Deviation ($\hat{\sigma}$)']
    kwargs = [{'cmap': cm.viridis,
               'vmax': 0.1},
              {'cmap': cm.coolwarm,
               'vmin': -5,
               'vmax': 5,
               'mask_color': 'black'}]
    for i in range(2):
        for k, obj in zip([0, 2], [raw_ins, OR_ins]):
            pl.image_plot(fig, ax[k, i], getattr(obj, attr[i])[:, 0, :, 0],
                          cbar_label=cbar_label[i],
                          freq_array=obj.freq_array[0], **kwargs[i])

    raw_ins.data[:, 0, :82] = np.ma.masked
    raw_ins.data[:, 0, -21:] = np.ma.masked
    mf = MF(raw_ins, sig_thresh=5, shape_dict=shape_dict, N_thresh=20)
    raw_ins.data_ms = raw_ins.mean_subtract(order=1)
    mf.apply_match_test(apply_N_thresh=True, order=1)
    raw_ins.data_ms = raw_ins.mean_subtract()
    for i in range(2):
        pl.image_plot(fig, ax[1, i], getattr(raw_ins, attr[i])[:, 0, :, 0],
                      cbar_label=cbar_label[i], freq_array=raw_ins.freq_array[0],
                      **kwargs[i])

    fig.savefig('%s/%s_flag_comp.png' % (figpath, obs))
    raw_ins.save()
    plt.close(fig)
