import numpy as np
import matplotlib.pyplot as plt
import plot_lib

# Location of data in the filesystem
arr_dir = '/Users/mike_e_dubs/MWA/Temperatures/Diffuse_2015_12s_Autos/RFI_Free/max_fit_arrays/'
plot_dir = '/Users/mike_e_dubs/MWA/Catalogs/Diffuse_2015_12s_Autos/Vis_Avg/Template/RFI_Free/Max_Fit_Corr/'

# Load in the arrays with the data
fit_coeff_array = np.load('%sfit_coeff_array.npy' % (arr_dir))
fit_centers_coeff_array = np.load('%sfit_centers_coeff_array.npy' % (arr_dir))
fit_edges_coeff_array = np.load('%sfit_edges_coeff_array.npy' % (arr_dir))
max_loc_array = np.load('%sbftemps.npy' % (arr_dir))

# Make plotting objects
fig_slope_main, ax_slope_main = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
fig_slope_centers, ax_slope_centers = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
fig_slope_edges, ax_slope_edges = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
fig_int_main, ax_int_main = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
fig_int_centers, ax_int_centers = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
fig_int_edges, ax_int_edges = plt.subplots(figsize=(14, 8), nrows=2, ncols=2)
fig_slope_main.suptitle('Main Slope vs. Beamformer Temperature')
fig_slope_centers.suptitle('Center Teeth Slope vs. Beamformer Temperature')
fig_slope_edges.suptitle('Edge Teeth Slope vs. Beamformer Temperature')
fig_int_main.suptitle('Main Intercept vs. Beamformer Temperature')
fig_int_centers.suptitle('Center Teeth Intercept vs. Beamformer Temperature')
fig_int_edges.suptitle('Edge Teeth Intercept vs. Beamformer Temperature')
pol_titles = ['XX', 'YY', 'XY', 'YX']

# Show bad obs on scatter plots
c = 30 * ['b']
# bad_obs_ind = [23, 24, 27, 28, 29]
# for k in bad_obs_ind:
#    c[k] = 'r'

for m in range(4):
    plot_lib.scatter_plot_2d(fig_slope_main, ax_slope_main[m / 2][m % 2],
                             max_loc_array, fit_coeff_array[:, 0, m],
                             title=pol_titles[m], xlabel='Beamformer Temperature (C)',
                             ylabel='Main Slope', c=c, ylim=[np.amin(fit_coeff_array[:, 0, m]),
                                                             np.amax(fit_coeff_array[:, 0, m])])
    plot_lib.scatter_plot_2d(fig_int_main, ax_int_main[m / 2][m % 2],
                             max_loc_array, fit_coeff_array[:, 1, m],
                             title=pol_titles[m], xlabel='Beamformer Temperature (C)',
                             ylabel='Main Y-Intercept', c=c)
    plot_lib.scatter_plot_2d(fig_slope_centers, ax_slope_centers[m / 2][m % 2],
                             max_loc_array, fit_centers_coeff_array[:, 0, m],
                             title=pol_titles[m], xlabel='Beamformer Temperature (C)',
                             ylabel='Center Teeth Slope', c=c, ylim=[np.amin(fit_centers_coeff_array[:, 0, m]),
                                                                     np.amax(fit_centers_coeff_array[:, 0, m])])
    plot_lib.scatter_plot_2d(fig_int_centers, ax_int_centers[m / 2][m % 2],
                             max_loc_array, fit_centers_coeff_array[:, 1, m],
                             title=pol_titles[m], xlabel='Beamformer Temperature (C)',
                             ylabel='Center Teeth Y-Intercept', c=c)
    plot_lib.scatter_plot_2d(fig_slope_edges, ax_slope_edges[m / 2][m % 2],
                             max_loc_array, fit_edges_coeff_array[:, 0, m],
                             title=pol_titles[m], xlabel='Beamformer Temperature (C)',
                             ylabel='Edge Teeth Slope', c=c, ylim=[np.amin(fit_edges_coeff_array[:, 0, m]),
                                                                   np.amax(fit_edges_coeff_array[:, 0, m])])
    plot_lib.scatter_plot_2d(fig_int_edges, ax_int_edges[m / 2][m % 2],
                             max_loc_array, fit_edges_coeff_array[:, 1, m],
                             title=pol_titles[m], xlabel='Beamformer Temperature (C)',
                             ylabel='Edge Teeth Y-Intercept', c=c)

fig_slope_main.savefig('%sslope_main_bf.png' % (plot_dir))
fig_int_main.savefig('%sint_main_bf.png' % (plot_dir))
fig_slope_centers.savefig('%sslope_centers_bf.png' % (plot_dir))
fig_int_centers.savefig('%sint_centers_bf.png' % (plot_dir))
fig_slope_edges.savefig('%sslope_edges_bf.png' % (plot_dir))
fig_int_edges.savefig('%sint_edges_bf.png' % (plot_dir))
