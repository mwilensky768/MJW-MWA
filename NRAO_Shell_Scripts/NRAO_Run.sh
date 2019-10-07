#!/bin/bash

while getopts ":f:o:y:r:i:" option
do
  case $option in
    # A text file where each line is an obsid
    f) obs_file_name="$OPTARG";;
    o) outdir=$OPTARG;;
    i) ins_file_name="$OPTARG";;
    y) yml_file_name="$OPTARG";;
    r) raw_dat_file_name="$OPTARG";;
    \?) echo "Unknown option: Accepted flags are -f (obs_file_name), -o (output directory), "
        echo "-q (python script to execute), -r (options for python script)."
        exit 1;;
    :) echo "Missing option argument for input flag"
       exit 1;;
  esac
done

#Manual shift to the next flag.
shift $(($OPTIND - 1))

#Throw error if no obs_id file.
if [ -z ${obs_file_name} ]; then
   echo "Need to specify a full filepath to a list of viable observation ids."
   exit 1
fi

#Throw error if no outdir.
if [ -z ${outdir} ]; then
   echo "Need to specify an output directory."
   exit 1
fi

N_obs=$(wc -l < $obs_file_name)

qsub -v obs_file_name=${obs_file_name},outdir=${outdir},ins_file_name=${ins_file_name},yml_file_name=${yml_file_name},raw_dat_file_name=${raw_dat_file_name} -q hera -j oe -o ${outdir} -l nodes=1:ppn=1 -l vmem=32G -t 1-${N_obs} -N SSINS /lustre/aoc/projects/hera/mwilensk/MJW-MWA/NRAO_Shell_Scripts/NRAO_job_task_array.sh
