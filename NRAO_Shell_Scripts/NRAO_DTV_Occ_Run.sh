#!/bin/bash

while getopts ":f:o:y:" option
do
  case $option in
    # A text file where each line is an obsid
    o) outdir=$OPTARG;;
    y) yml_file_name="$OPTARG";;
    \?) echo "Unknown option: Accepted flags are -f (obs_file_name), -o (output directory), "
        echo "-q (python script to execute), -r (options for python script)."
        exit 1;;
    :) echo "Missing option argument for input flag"
       exit 1;;
  esac
done

#Manual shift to the next flag.
shift $(($OPTIND - 1))

#Throw error if no outdir.
if [ -z ${outdir} ]; then
   echo "Need to specify an output directory."
   exit 1
fi

#Throw error if no yml_file_name.
if [ -z ${yml_file_name} ]; then
   echo "Need to specify a file with paths to yml files."
   exit 1
fi

qsub -v outdir=${outdir},yml_file_name=${yml_file_name} -q hera -j oe -o ${outdir} -l nodes=1:ppn=1 -l vmem=10G -N SSINS_DTV_occ /lustre/aoc/projects/hera/mwilensk/MJW-MWA/NRAO_Shell_Scripts/NRAO_DTV_Hist_Job.sh
