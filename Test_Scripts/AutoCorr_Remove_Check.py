import rfipy as rfi
import numpy as np
import matplotlib.pyplot as plt

bad_blt_inds = np.load('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/bad_blt_inds.npy')
bad_blt_inds = bad_blt_inds.tolist()

RFI = rfi.RFI('1061318864', '/Users/mike_e_dubs/python_stuff/uvfits/1061318864.uvfits',
              bad_blt_inds=bad_blt_inds)


flag_slices = ['Unflagged', 'All']

data = []
for flag_slice in flag_slices:
    data.append(RFI.one_d_hist_prepare(flag_slice=flag_slice))

fig, ax = plt.subplots(figsize=(14, 8))

RFI.one_d_hist_plot(fig, ax, data, flag_slices, '1061318864 Autocorr + time edges remove')

plt.savefig('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/1061318864_Autocorr_TimeEdge_Remove_1DHist.png')
plt.show()
