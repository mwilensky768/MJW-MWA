#!/bin/bash

unset obs_file_name
unset outdir

while getopts ":f:o:s:m:t:" option
do
  case $option in
    f) obs_file_name="$OPTARG";;
    o) outdir=$OPTARG;;
    s) script=$OPTARG;;
    m) mem=$OPTARG;;
    t) wall=$OPTARG;;
  esac
done

while read obs
do
  sbatch --nodes=1 --mem=$mem --time=$wall --account=mwaeor --job-name=SSINS --output=SSINS_%j.out --error=SSINS_%j.e --export=obs=$obs,outdir=$outdir $script
done < $obs_file_name
