pro fhd_flag_read, flag_file

  restore, flag_file
  im_xx = TOTAL(TRANSPOSE(REFORM(TRANSPOSE(*VIS_WEIGHTS[0]), [8128, 7, 80])), 3)
  im_yy = TOTAL(TRANSPOSE(REFORM(TRANSPOSE(*VIS_WEIGHTS[1]), [8128, 7, 80])), 3)

  quick_image, im_xx, INDGEN(80), INDGEN(7)
