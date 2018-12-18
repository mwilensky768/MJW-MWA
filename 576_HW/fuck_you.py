from SSINS import SS, plot_lib
import numpy as np
import matplotlib.pyplot as plt

freq_chans = np.load('/Volumes/Faramir/uvfits/freq_chans.npy')
ss = SS(obs='1066742016', inpath='/Volumes/Faramir/uvfits/1066742016.uvfits',
        outpath='/Users/mikewilensky/576/1066742016', read_kwargs={'file_type': 'uvfits',
                                                                   'ant_str': 'cross',
                                                                   'freq_chans': freq_chans})

freq_chans = np.load('/Volumes/Faramir/uvfits/freq_chans.npy')
counts_1, bins_1 = np.histogram(ss.UV.data_array[2:21], bins='auto')
counts_2, bins_2 = np.histogram(ss.UV.data_array[21:-4], bins='auto')

counts_1 = np.append(counts_1, 0)
counts_2 = np.append(counts_2, 0)

fig, ax = plt.subplots()
plot_lib.error_plot(fig, ax, bins_1, counts_1, label='Contaminated Times', yscale='log',
                    xscale='linear', drawstyle='steps-post')
plot_lib.error_plot(fig, ax, bins_2, counts_2, label='Clean Times', yscale='log',
                    xscale='linear', drawstyle='steps-post')

fig.savefig('/Users/mikewilensky/fuck_you.png')
