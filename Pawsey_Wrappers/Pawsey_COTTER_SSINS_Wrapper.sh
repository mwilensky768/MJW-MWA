#!/bin/bash

unset obs_file_name
unset outdir

while getopts ":f:o:s:m:t:tr:fr:e:i:ef" option
do
  case $option in
    f) obs_file_name="$OPTARG";;
    o) outdir=$OPTARG;;
    s) script=$OPTARG;;
    m) mem=$OPTARG;;
    t) wall=$OPTARG;;
    tr) timeres=$OPTARG;;
    fr) freqres=$OPTARG;;
    e) edgewidth=$OPTARG;;
    i) initflag=$OPTARG;;
    ef) endflag=$OPTARG;;
  esac
done

while read obs
do
  sbatch --nodes=1 --mem=$mem --time=$wall --account=mwaeor --job-name=SSINS --output=SSINS_%j.out --error=SSINS_%j.e --export=obs=$obs,outdir=$outdir,timeres=$timeres,freqres=$freqres,edgewidth=$edgewidth,initflag=$initflag,endflag=$endflag $script
done < $obs_file_name
