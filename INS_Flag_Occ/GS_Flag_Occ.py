import numpy as np
import glob
from SSINS import plot_lib as pl
import os
import matplotlib.pyplot as plt

basedir = '/Users/mike_e_dubs/MWA/INS'
COTTER_Nbls_arrs = glob.glob('%s/Golden_Set/arrs/*_original_INS_Nbls.npym' % (basedir))
total_Nbls_arrs = glob.glob('%s/Golden_Set/arrs/*None_INS_Nbls.npym' % (basedir))
data_arrs = glob.glob('%s/Golden_Set_Filtered_5s_no_Streak/arrs/*data_ms*.npym' % (basedir))
outdir = '%s/Golden_Set_Total_Occupancy/No_Streak' % basedir
freqs = np.load('%s/Golden_Set/metadata/1061311664_freq_array.npy' % basedir)[0]
freqs = np.array(['%.2f' % (freq * 10 ** (-6)) for freq in freqs]).astype(float)
if not os.path.exists(outdir):
    os.makedirs(outdir)

ledges = [0 + 16 * i for i in range(24)]
redges = [15 + 16 * i for i in range(24)]

for dat in [COTTER_Nbls_arrs, total_Nbls_arrs, data_arrs]:
    dat.sort()

for i, (COTTER_arr, total_arr, data_arr) in enumerate(zip(COTTER_Nbls_arrs, total_Nbls_arrs, data_arrs)):
    COTTER_Nbls = np.ma.masked_array(np.load(COTTER_arr).astype(float))
    total_Nbls = np.ma.masked_array(np.load(total_arr).astype(float))
    data = np.load(data_arr)
    SSINS_Nbls = total_Nbls * data.mask
    COTTER_Nbls[:, :, ledges + redges] = np.ma.masked
    total_Nbls[:, :, ledges + redges] = np.ma.masked
    SSINS_Nbls[:, :, ledges + redges] = np.ma.masked
    if not i:
        total_occ_COTTER_num = np.ma.copy(total_Nbls - COTTER_Nbls)
        total_occ_den = np.ma.copy(total_Nbls)
        total_occ_SSINS_num = np.ma.copy(SSINS_Nbls)
    else:
        total_occ_COTTER_num += total_Nbls - COTTER_Nbls
        total_occ_den += total_Nbls
        total_occ_SSINS_num += SSINS_Nbls

total_occ_COTTER_wf = total_occ_COTTER_num / total_occ_den
total_occ_SSINS_wf = total_occ_SSINS_num / total_occ_den

total_occ_COTTER_freq = total_occ_COTTER_num.sum(axis=(0, 3)) / total_occ_den.sum(axis=(0, 3)) * 100
total_occ_SSINS_freq = total_occ_SSINS_num.sum(axis=(0, 3)) / total_occ_den.sum(axis=(0, 3)) * 100
total_occ_COTTER = total_occ_COTTER_num.sum() / total_occ_den.sum() * 100
total_occ_SSINS = total_occ_SSINS_num.sum() / total_occ_den.sum() * 100
print('COTTER occupancy is %f%%' % total_occ_COTTER)
print('SSINS occupancy is %f%%' % total_occ_SSINS)

fig, ax = plt.subplots(figsize=(14, 8))
labels = ['COTTER', 'SSINS']
for i, dat in enumerate([total_occ_COTTER_freq, total_occ_SSINS_freq]):
    pl.error_plot(fig, ax, freqs, dat, xlabel='Frequency (Mhz)',
                  ylabel='Occupancy (%)', label=labels[i], drawstyle='default',
                  title='Golden Set RFI Occupancy SSINS vs. COTTER')
fig.savefig('%s/COTTER_SSINS_Spectral_Occupancy_Edges_Careful_Mask.png' % outdir)
