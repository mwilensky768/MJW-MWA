import numpy as np
import matplotlib.pyplot as plt
import plot_lib
import glob
from matplotlib.ticker import AutoMinorLocator

GC_path = '/Users/mike_e_dubs/MWA/Catalogs/Grand_Catalog/'
survey = 'S2_Zenith_8s_Autos/'
flag_slice = ''
freq_array_path = '/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/MWA_Highband_Freq_Array.npy'

arr_dir = '%s%sVis_Avg/%sAverages/' % (GC_path, survey, flag_slice)
fig_dir = '%s%sVis_Avg/%s' % (GC_path, survey, flag_slice)


arr_path_list = glob.glob('%s*.npy' % (arr_dir))


freq_array = np.load(freq_array_path)

pols = ['XX', 'YY', 'XY', 'YX']
Nfreqs = len(freq_array)
xticks = [Nfreqs * k / 6 for k in range(6)]
xticks.append(Nfreqs - 1)
xminors = AutoMinorLocator(4)
yminors = 'auto'
xticklabels = ['%.1f' % (freq_array[tick] * 10 ** (-6)) for tick in xticks]


for path in arr_path_list:
    INS = np.load(path)
    obs = path[path.find('Averages/') + 9:path.find('Averages/') + 19]
    fig, ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
    fig.suptitle('%s Incoherent Noise Spectrum, All Baselines' % (obs))

    for m in range(4):

        plot_lib.image_plot(fig, ax[m / 2][m % 2], INS[:, 0, :, m], title=pols[m],
                            cbar_label='UNCALIB', xticks=xticks, xminors=xminors,
                            yminors=yminors, xticklabels=xticklabels, zero_mask=False,
                            invalid_mask=True)

    fig.savefig('%s%s_INS_All.png' % (fig_dir, obs))
    plt.close(fig)
