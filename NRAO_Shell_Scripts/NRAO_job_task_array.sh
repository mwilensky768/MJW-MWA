#!/bin/bash

echo JOBID ${PBS_JOBID}
echo TASKID ${PBS_ARRAYID}
echo "JOB START TIME" `date +"%Y-%m-%d_%H:%M:%S"`

# Use the obsid of the arrayid'th line of the obs_file
obs_id=$(sed "${PBS_ARRAYID}q;d" ${obs_file_name})

echo "Processing $obs_id"

#strip the last / if present in output directory filepath
outdir=${outdir%/}
echo Using output directory: $outdir

# Run python catalog script
python $script $script_args
