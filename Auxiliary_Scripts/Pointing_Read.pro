pro Pointing_Read

  pathlist = '/nfs/eor-00/h1/mwilensk/Grand_Catalog/Golden_Set_8s_Autos/Golden_Set_OBSIDS_paths.txt'
  
  ; Select a text file and open for reading
  file = DIALOG_PICKFILE(FILTER=pathlist)
  OPENR, lun, file, /GET_LUN
  ; Read one line at a time, saving the result into array
  array = ''
  line = ''
  WHILE NOT EOF(lun) DO BEGIN & $
    READF, lun, line & $
    array = [array, line] & $
  ENDWHILE
  ; Close the file and free the file unit
  FREE_LUN, lun
  
  delays = ''
  
  for k=1,94,1 do begin
    path = array[k]
    strings = HEADFITS(array[k])
    delays = [delays, strings[154]]
  endfor
  
  PRINT, delays
  
  fname='/nfs/eor-00/h1/mwilensk/Grand_Catalog/Golden_Set_8s_Autos/Golden_Set_Delays.txt'
  OPENW,1,fname
  PRINTF,delays,FORMAT='(A64)'
  CLOSE,1
  
  
END
