#!/bin/bash -l

echo JOBID ${PBS_JOBID}
echo TASKID ${PBS_ARRAYID}
echo "JOB START TIME" `date +"%Y-%m-%d_%H:%M:%S"`

# Use the obsid of the arrayid'th line of the obs_file
obs_id=$(sed "${PBS_ARRAYID}q;d" ${obs_file_name})
raw_file=$(sed "${PBS_ARRAYID}q;d" ${raw_dat_file_name})
ins_file=$(sed "${PBS_ARRAYID}q;d" ${ins_file_name})
yml_file=$(sed "${PBS_ARRAYID}q;d" ${yml_file_name})

echo "Processing $obs_id"

#strip the last / if present in output directory filepath
outdir=${outdir%/}
echo Using output directory: $outdir

echo $(which python)


# Run python catalog scripts
/lustre/aoc/projects/hera/mwilensk/anaconda2/bin/python /lustre/aoc/projects/hera/mwilensk/MJW-MWA/HERA_RFI/DTV_stat.py -i $ins_file -y $yml_file -r $raw_file -o ${outdir}/${obs_id}_DTV_occ_dict.yml
