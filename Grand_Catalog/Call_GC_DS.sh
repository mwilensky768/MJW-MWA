#! /bin/bash
#$ -V
#$ -N GC_DS
#$ -S /bin/bash

echo JOBID ${JOB_ID}
echo TASKID ${SGE_TASK_ID}

python /nfs/eor-00/h1/mwilensk/MJW-MWA/GC_DS.py ${SGE_TASK_ID}
