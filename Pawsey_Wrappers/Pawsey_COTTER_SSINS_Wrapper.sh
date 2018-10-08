#!/bin/bash

unset obs_file_name
unset outdir

while getopts ":f:o:" option
do
  case $option in
    f) obs_file_name="$OPTARG";;
    o) outdir=$OPTARG;;
  esac
done

while read obs
do
  for box_num in 0{1..9} {10..24}
  do
    sbatch --nodes=1 --mem=60000 --time=12:00:00 --account=mwaeor --job-name=SSINS --output=SSINS_%j.out --error=SSINS_%j.e --export=obs=$obs,outdir=$outdir,box_num=$box_num /home/mwilensky/MJW-MWA/Pawsey_Wrappers/Pawsey_Catalog_Run.sh
  done
done < $obs_file_name
