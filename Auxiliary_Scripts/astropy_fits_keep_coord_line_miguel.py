plt.figure(figsize=(10,10))

#A cut area. Currently in pixel units from 20148x20148 original
pix_left = 512
pix_right = 1536
pix_bottom = 0
pix_top = 512

#key of aligning coordinates and data is to cut them both
data_cut = hdu[0].data[pix_bottom:pix_top, pix_left:pix_right]
wcs_cut = wcs[pix_bottom:pix_top, pix_left:pix_right]

ax = plt.subplot(projection=wcs_cut)
plt.imshow(data_cut, vmin=-.2, vmax=.2, origin=‘lower’,cmap=‘gray’)
lon = ax.coords[0]
lat = ax.coords[1]

lon.set_ticks(spacing=30 * units.degree)
lat.set_ticks(spacing=15. * units.degree)
plt.grid(color=‘blue’, ls=‘solid’, alpha=0.5)
