import rfipy as rfi
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

RFI = rfi.RFI()

RFI.read_even_odd('/Users/mike_e_dubs/python_stuff/uvfits/1061313008.uvfits')

W, unique_times = RFI.waterfall_hist_prepare((2 * 10**3, 5 * 10**5), plot_type='ant-freq',
                                             fraction=False, flag_slice='All')

N_events = W.shape[3]

gs = GridSpec(4, 1)

figs = [plt.figure(figsize=(14, 8)) for k in range(N_events)]
axes = [[figs[k].add_subplot(gs[l, 0]) for l in range(4)] for k in range(N_events)]
vmax = [[np.amax(W[:, :, l, k]) for l in range(4)] for k in range(N_events)]

pols = ['XX', 'YY', 'XY', 'YX']

for k in range(N_events):
    for l in range(RFI.UV.Npols):
        RFI.waterfall_hist_plot(figs[k], axes[k][l], W[:, :, l, k], 'Drill ' + pols[l] + ' t = ' +
                                str(unique_times[k]), vmax[k][l], aspect_ratio=1, fraction=False)
    figs[k].savefig('/Users/mike_e_dubs/python_stuff/RFI_Diagnostic/drill_test_' + str(k))
    plt.close(figs[k])
