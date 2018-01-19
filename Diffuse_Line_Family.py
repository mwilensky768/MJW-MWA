import numpy as np
import matplotlib.pyplot as plt
import plot_lib

arr_dir = '/Users/mike_e_dubs/MWA/Temperatures/Diffuse_2015_12s_Autos/RFI_Free/max_fit_arrays/'
freq_array_path = '/Users/mike_e_dubs/python_stuff/MJW-MWA/Useful_Information/MWA_Highband_Freq_Array.npy'
plot_path = '/Users/mike_e_dubs/MWA/Catalogs/Diffuse_2015_12s_Autos/Vis_Avg/Template/RFI_Free/Max_Fit_Corr/'
freqs = np.load(freq_array_path)

# Load in the arrays with the data
fit_coeff_array = np.load('%sfit_coeff_array.npy' % (arr_dir))
fit_centers_coeff_array = np.load('%sfit_centers_coeff_array.npy' % (arr_dir))
fit_edges_coeff_array = np.load('%sfit_edges_coeff_array.npy' % (arr_dir))

# Initialize the fit arrays...
fit = np.zeros([len(freqs), 4])
fit_centers = np.zeros([len(freqs), 4])
fit_edges = np.zeros([len(freqs), 4])

# Initialize the plotting figures
fig_main, ax_main = plt.subplots(figsize=(14, 8), ncols=2, nrows=2)
fig_center, ax_center = plt.subplots(figsize=(14, 8), ncols=2, nrows=2)
fig_edge, ax_edge = plt.subplots(figsize=(14, 8), ncols=2, nrows=2)
fig_center_ratio, ax_center_ratio = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
fig_edges_ratio, ax_edges_ratio = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)

fig_main.suptitle('Main Body Fits')
fig_center.suptitle('Center Spike Fits')
fig_edge.suptitle('Edge Dip Fits')
fig_center_ratio.suptitle('Center Spike to Main Body Fit Ratio')
fig_edges_ratio.suptitle('Edge Dip to Main Body Fit Ratio')

pols = ['XX', 'YY', 'XY', 'YX']

for n in range(fit_coeff_array.shape[0]):
    for m in range(fit_coeff_array.shape[2]):
        fit[:, m] = np.sum(np.array([fit_coeff_array[n, k, m] * freqs ** (1 - k)
                                     for k in range(fit_coeff_array.shape[1])]),
                           axis=0)

        fit_centers[:, m] = np.sum(np.array([fit_centers_coeff_array[n, k, m] *
                                             freqs ** (1 - k) for k in
                                             range(fit_coeff_array.shape[1])]),
                                   axis=0)

        fit_edges[:, m] = np.sum(np.array([fit_edges_coeff_array[n, k, m] *
                                           freqs ** (1 - k) for k in
                                           range(fit_coeff_array.shape[1])]),
                                 axis=0)

        plot_lib.line_plot(fig_main, ax_main[m / 2][m % 2], [fit[:, m], ],
                           title=pols[m], xlabel='Frequency Channel #',
                           ylabel='Visibility Amplitude (UNCALIB)',
                           zorder=[1, ], labels=['', ], legend=False)
        plot_lib.line_plot(fig_center, ax_center[m / 2][m % 2], [fit_centers[:, m], ],
                           title=pols[m], xlabel='Frequency Channel #',
                           ylabel='Visibility Amplitude (UNCALIB)',
                           zorder=[1, ], labels=['', ], legend=False)
        plot_lib.line_plot(fig_edge, ax_edge[m / 2][m % 2], [fit_edges[:, m], ],
                           title=pols[m], xlabel='Frequency Channel #',
                           ylabel='Visibility Amplitude (UNCALIB)',
                           zorder=[1, ], labels=['', ], legend=False)
        plot_lib.line_plot(fig_center_ratio, ax_center_ratio[m / 2][m % 2],
                           [fit_centers[:, m] / fit[:, m], ],
                           title=pols[m], xlabel='Frequency Channel #',
                           ylabel='Ratio', zorder=[1, ], labels=['', ],
                           legend=False)
        plot_lib.line_plot(fig_edges_ratio, ax_edges_ratio[m / 2][m % 2],
                           [fit_edges[:, m] / fit[:, m], ],
                           title=pols[m], xlabel='Frequency Channel #',
                           ylabel='Ratio', zorder=[1, ], labels=['', ],
                           legend=False)

fig_main.savefig('%smain_fits_family.png' % (plot_path))
fig_center.savefig('%scenter_fits_family.png' % (plot_path))
fig_edge.savefig('%sedge_fits_family.png' % (plot_path))
fig_center_ratio.savefig('%scenter_fit_ratio_family.png' % (plot_path))
fig_edges_ratio.savefig('%sedges_fit_ratio_family.png' % (plot_path))
