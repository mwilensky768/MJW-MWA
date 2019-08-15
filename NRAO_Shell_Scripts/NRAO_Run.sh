#!/bin/bash

while getopts ":f:d:o:q:r:i:" option
do
  case $option in
    # A text file where each line is an obsid
    f) obs_file_name="$OPTARG";;
    o) outdir=$OPTARG;;
    q) script=$OPTARG;;
    r) script_args=$OPTARG;;
    i) indir=$OPTARG;;
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

#Throw error if no script.
if [ -z ${script} ]; then
   echo "Need to specify an script."
   exit 1
fi

#Throw error if no indir.
if [ -z ${indir} ]; then
   echo "Need to specify an input directory."
   exit 1
fi

N_obs=$(wc -l < $obs_file_name)

qsub -V -q hera -j -o ${outdir} -l nodes=10:ppn=4 -t 1:${N_obs} -N SSINS /lustre/aoc/projects/hera/mwilensk/MJW-MWA/NRAO_Shell_Scripts/NRAO_job_task_array.sh &
