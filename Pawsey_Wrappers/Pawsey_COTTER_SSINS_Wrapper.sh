#!/bin/bash

unset obs_file_name
unset outdir

while getopts ":f:o:s:m:t:a:b:e:i:c" option
do
  case $option in
    f) obs_file_name="$OPTARG";;
    o) outdir=$OPTARG;;
    s) script=$OPTARG;;
    m) mem=$OPTARG;;
    t) wall=$OPTARG;;
    a) timeres=$OPTARG;;
    b) freqres=$OPTARG;;
    e) edgewidth=$OPTARG;;
    i) initflag=$OPTARG;;
    c) endflag=$OPTARG;;
  esac
done

N_obs=$(wc -l < $obs_file_name)

sbatch --array=1-$N_obs --nodes=1 --mem=$mem --time=$wall --account=mwaeor --job-name=SSINS --output=SSINS_%j.out --error=SSINS_%j.e --export=ALL,obs_file_name=${obs_file_name},outdir=${outdir},timeres=${timeres},freqres=${freqres},edgewidth=${edgewidth},initflag=${initflag},endflag=${endflag} $script
