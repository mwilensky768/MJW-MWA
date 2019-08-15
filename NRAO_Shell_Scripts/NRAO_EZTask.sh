#!/bin/bash

echo JOBID ${PBS_JOBID}
echo TASKID ${PBS_ARRAYID}
echo "JOB START TIME" `date +"%Y-%m-%d_%H:%M:%S"`

# Use the obsid of the arrayid'th line of the obs_file
obs_id=$(sed "${PBS_ARRAYID}q;d" ${obs_file_name})

echo "Processing $obs_id"

source /users/iware/.bashrc

conda activate hera
which python

python /lustre/aoc/projects/hera/iware/CHAMP/ElCap_Desktop/RFI_project/SSINS_code_IW.py -d $obs_id

echo "JOB END TIME" `date +"%Y-%m-%d_%H:%M:%S"`
