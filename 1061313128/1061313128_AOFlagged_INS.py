from SSINS import SS, plot_lib
import matplotlib.pyplot as plt
from matplotlib import cm

obs = '1061313128'
inpath = '/Users/mike_e_dubs/MWA/Data/uvfits/1061313128.uvfits'
inpath2 = '/Users/mike_e_dubs/MWA/Data/uvfits/1061313128_noflag.uvfits'

ss = SS(obs=obs, inpath=inpath, bad_time_indices=[0, -1, -2, -3],
        read_kwargs={'ant_str': 'cross'}, flag_choice='original')
ss.INS_prepare()
fig, ax = plt.subplots(figsize=(14, 8), nrows=2)
plot_lib.image_plot(fig, ax[1], ss.INS.data_ms[:, 0, :, 0], cmap=cm.coolwarm,
                    cbar_label='Deviation ($\hat{\sigma}$)',
                    title='MWA Highly Contaminated (AOFlagger Applied)',
                    freq_array=ss.UV.freq_array[0])
del ss
ss2 = SS(obs=obs, inpath=inpath2, bad_time_indices=[0, -1, -2, -3],
         read_kwargs={'ant_str': 'cross'}, flag_choice=None)
ss2.INS_prepare()

plot_lib.image_plot(fig, ax[0], ss2.INS.data_ms[:, 0, :, 0], cmap=cm.coolwarm,
                    cbar_label='Deviation ($\hat{\sigma}$)',
                    title='MWA Highly Contaminated (No Flagging)',
                    freq_array=ss2.UV.freq_array[0])
fig.savefig('/Users/mike_e_dubs/General/%s/figs/%s_Flag_Compare.png' % (obs, obs))
