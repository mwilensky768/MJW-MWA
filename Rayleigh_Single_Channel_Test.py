import rfipy as rfi
import matplotlib.pyplot as plt

obs = '1061313008'
inpath = '/Users/mike_e_dubs/python_stuff/uvfits/' + obs + '.uvfits'
freq_drill = 300
label = 'Unflagged'

RFI = rfi.RFI(obs, inpath)

f = RFI.UV.freq_array[0, freq_drill]
title = obs + ' f = ' + str(f) + ' Rayleigh Test'

data = RFI.one_d_hist_prepare(freq_drill=freq_drill)

fig, ax = plt.subplots()

RFI.one_d_hist_plot(fig, ax, data, label, title, normed=False, fit=True, fit_window=[1, 1000])

fig.savefig('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/1061313008_Rayleigh_Test_f300.png')

plt.show()
