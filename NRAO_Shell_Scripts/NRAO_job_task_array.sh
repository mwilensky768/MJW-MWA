#!/bin/bash

echo JOBID ${JOB_ID}
echo TASKID ${SGE_TASK_ID}
echo "JOB START TIME" `date +"%Y-%m-%d_%H:%M:%S"`

obs_id=$(sed "${SGE_TASK_ID}q;d" ${obs_file_name})

echo "Processing $obs_id"

#strip the last / if present in output directory filepath
outdir=${outdir%/}
echo Using output directory: $outdir

# Run python catalog script
python $script ${obs_id} ${indir}/${obs_id}.uvh5 $outdir $script_args
