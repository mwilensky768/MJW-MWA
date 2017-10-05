import rfipy as rfi
import matplotlib.pyplot as plt

obs = '1061313008'
inpath = '/Users/mike_e_dubs/python_stuff/uvfits/' + obs + '.uvfits'
freq_drill = []

RFI = rfi.RFI(obs, inpath, bad_time_indices=[0, -3, -2, -1], filetype='uvfits')

f = RFI.UV.freq_array[0, freq_drill]
title = obs + ' f = ' + str(f) + ' hz Rayleigh Test'

data = RFI.one_d_hist_prepare(freq_drill=freq_drill)

fig, ax = plt.subplots(nrows=2, figsize=(14, 8))

RFI.one_d_hist_plot(fig, ax[0], data, title, normed=False, fit=True, res_ax=ax[1],
                    fit_window=[0, 10**(12)])

fig.savefig('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/' + obs +
            '_Rayleigh_Test_NoCut_LLN_Residual.png')

plt.show()
