`from astropy.io import fits`
`from astropy.wcs import WCS`
`from astropy import units` (edited)

`hdu = fits.open(rfi_file)`
`wcs = WCS(hdu[0].header)`
`plt.figure(figsize=(10,10))`
`ax = plt.subplot(projection=wcs)`
`plt.imshow(hdu[0].data, vmin=-.2, vmax=.2, origin='lower', cmap='gray')`
`lon = ax.coords[0]`
`lat = ax.coords[1]`

`lon.set_ticks(spacing=30. * units.degree)`
`lat.set_ticks(spacing=15. * units.degree)`
`plt.grid(color='blue', ls='solid', alpha=0.3)`
