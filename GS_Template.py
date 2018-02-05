import numpy as np
import matplotlib.pyplot as plt
import plot_lib

rfi_list_file = '/Users/mike_e_dubs/MWA/Obs_Lists/GS_8s_Autos_INS_RFI.txt'
gs_list_file = '/Users/mike_e_dubs/MWA/Obs_Lists/Golden_Set_OBSIDS.txt'
INS_array_path = '/Users/mike_e_dubs/MWA/GC/Grand_Catalog/Golden_Set_8s_Autos/Vis_Avg/Averages/'
freq_array_path = '/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/'

with open(rfi_list_file) as f:
    rfi_list = f.read().split("\n")
with open(gs_list_file) as g:
    obs_list = g.read().split("\n")

for obs in rfi_list:
    obs_list.remove(obs)

fig, ax = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
pols = ['XX', 'YY', 'XY', 'YX']

for obs in obs_list:
    INS = np.load('%s%s_Vis_Avg_Amp_All.npy' % (INS_array_path, obs))[:, 0, :, :]
    mean = np.mean(INS, axis=0)
    template = np.array([mean for k in range(INS.shape[0])])

    Excess = INS - template
    Frac_Diff = Excess / template

    fig.suptitle('%s Incoherent Noise Spectrum Excess')
    for m in range(4):
        plot_lib.image_plot(fig, ax[m / 2][m % 2], Frac_Diff[:, :, m],
                            cmap=cm.coolwarm, title=pols[m], cbar_label='Fraction',
                            xticks=[])
