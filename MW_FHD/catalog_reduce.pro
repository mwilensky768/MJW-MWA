pro reduce_skymodel, skymodel, flux_keep, outpath

  restore, catalog_path
  fluxes = skymodel.source_list.flux[0].i
  sort_inds = sort(fluxes)
  ; These are in ascending order
  sorted_fluxes = fluxes[sort_inds]

  ; Get rid of negative values
  wh_neg = where(sorted_fluxes le 0, num_neg)
  if num_neg gt 0 then begin
    ; Already sorted, so no need to figure out more indices
    start_ind = max(wh_neg) + 1
    end_ind = n_elements(sorted_fluxes) - 1
    sorted_fluxes = sorted_fluxes[start_ind:end_ind]
    sort_inds = sort_inds[start_ind:end_ind]
  endif

  cuflux = total(sorted_fluxes, /cumulative)
  total_flux = total(sorted_fluxes)
  cufrac = cuflux / total_flux

  ; Find the highest index to keep the desired flux fraction
  wh_cf_gt_fk = where(cufrac gt flux_keep)
  max_ind = min(wh_cf_gt_fk) - 1

  inds_keep = sort_inds[0:max_ind]
  max_flux_level_keep = sorted_fluxes[max_ind]
  catalog = skymodel.source_list
  catalog = catalog[inds_keep]

  save, catalog, filename=outpath
