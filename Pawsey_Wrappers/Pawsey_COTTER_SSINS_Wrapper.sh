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

data_dir=/astro/mwaeor/MWA/data
module use /group/mwa/software/modulefiles
module load numpy/1.15.1
module load scipy/1.1.0
module load matplotlib/2.2.3
module load h5py/2.8.0
module load six
module load pyuvdata/master
module load cotter

while read obs
do
  sbatch -nnodes=1 --ntasks=24 --time=06:00:00 --account=mwaeor --job-name=SSINS --output=SSINS_%j.out error=SSINS_%j.e --export=ALL Pawsey_Catalog_Run.sh
done < $obs_file_name
