import rfipy as rfi
import matplotlib.pyplot as plt

obs = '1061313008'
inpath = '/Users/mike_e_dubs/python_stuff/uvfits/' + obs + '.uvfits'
title = '1061313008 Rayleigh Superposition Test'
fig, ax = plt.subplots(nrows=2, figsize=(14, 8))
outpath = '/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/'

RFI = rfi.RFI(obs, inpath)

data = RFI.one_d_hist_prepare(fit=True)

fig, ax = plt.subplots(nrows=2, figsize=(14, 8))

RFI.one_d_hist_plot(fig, ax[0], data, title, res_ax=ax[1])

fig.savefig(outpath + obs + '_Rayleigh_Superposition.png')
