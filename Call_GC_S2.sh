#! /bin/bash
#$ -V
#$ -N GS_RFI
#$ -S /bin/bash

echo JOBID ${JOB_ID}
echo TASKID ${SGE_TASK_ID}

python /nfs/eor-00/h1/mwilensk/MJW-MWA/GC_S2.py ${SGE_TASK_ID}
