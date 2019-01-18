from SSINS import SS, plot_lib
from pyuvdata import UVData
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

indir = '/Volumes/Faramir/uvfits'
obslist= ['1061312640', '1066742016']

for obs in obslist:
    UV = UVData()
    UV.read('%s/%s.uvfits' (indir, obs), file_type='uvfits', polarizations=-5, obs=obs)
    UV.select(times=np.unique(UV.time_array)[1:-3], ant_str='cross')
    ss = SS(UV=UV, outpath='/Users/mikewilensky/SSINS_Paper')
    ss.INS_prepare()
    fig, ax = plt.subplots(figsize=(16, 9))
    plot_lib.image_plot(fig, ax, ss.INS.data[:, 0, :, 0], aspect='auto',
                        freq_array=UV.freq_array[0], ylabel='Time (2 s)',
                        xlabel='Frequency (Mhz)')
    fig_ms, ax_ms = plt.subplots(figsize=(16, 9))
    plot_lib.image_plot(fig, ax, ss.INS.data_ms[:, 0, :, 0], aspect='auto',
                        freq_array=UV.freq_array[0], ylabel='Time (2 s)',
                        xlabel='Frequency (Mhz)', cmap=cm.coolwarm)
    fig.savefig('%s/%s_INS_data.pdf' % (ss.outpath, obs))
    fig_ms.savefig('%s/%s_INS_data_ms.pdf' % (ss.outpath, obs))
    plt.close(fig)
    plt.close(fig_ms)
    del ss
