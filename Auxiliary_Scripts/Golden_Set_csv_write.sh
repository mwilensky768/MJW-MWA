#!/bin/bash

file=/Users/mike_e_dubs/python_stuff/MJW-MWA/Obs_Lists/ASVO_GS.csv
obsfile=/Users/mike_e_dubs/python_stuff/MJW-MWA/Obs_Lists/Golden_Set_OBSIDS.txt

while read obs
do
  echo "obs_id=${obs}, job_type=c, timeres=0.5, freqres=40, edgewidth=80, conversion=uvfits, flagdcchannels=false, norfi=true" >> $file
done < $obsfile
